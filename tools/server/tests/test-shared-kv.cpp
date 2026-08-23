#include "server-shared-kv.h"

#undef NDEBUG
#include <cassert>
#include <vector>

static void test_checkpoint_caps() {
    constexpr int32_t configured_cap = 32;
    constexpr int32_t keeper_slot    = 1;

    assert(server_effective_checkpoint_cap(configured_cap, keeper_slot, keeper_slot) == configured_cap);
    assert(server_effective_checkpoint_cap(configured_cap, keeper_slot, 0) == 0);
    assert(server_effective_checkpoint_cap(configured_cap, keeper_slot, 2) == 0);

    assert(server_effective_checkpoint_cap(configured_cap, -1, 0) == configured_cap);
    assert(server_effective_checkpoint_cap(-1, keeper_slot, keeper_slot) == 0);
}

static void test_slot_ids() {
    assert(server_slot_id_is_valid(-1, 4, true));
    assert(!server_slot_id_is_valid(-1, 4, false));
    assert(server_slot_id_is_valid(0, 4, false));
    assert(server_slot_id_is_valid(3, 4, false));
    assert(!server_slot_id_is_valid(4, 4, false));
    assert(!server_slot_id_is_valid(-2, 4, true));
}

static void test_checkpoint_retention() {
    std::vector<int> mirror_checkpoints = { 1, 2, 3 };
    server_apply_checkpoint_cap(mirror_checkpoints, 0);
    assert(mirror_checkpoints.empty());

    std::vector<int> keeper_checkpoints = { 1, 2, 3 };
    server_apply_checkpoint_cap(keeper_checkpoints, 32);
    assert(keeper_checkpoints.size() == 3);

    server_apply_checkpoint_cap(keeper_checkpoints, 2);
    assert((keeper_checkpoints == std::vector<int>{ 2, 3 }));
}

static void test_expectations() {
    const server_shared_kv_observation observation = {
        /* donor_slot           */ 2,
        /* adopted_prefix_length */ 8192,
    };

    assert(server_shared_kv_check_expectation({}, observation) == server_shared_kv_mismatch::none);

    server_shared_kv_expectation matching;
    matching.donor_slot            = 2;
    matching.adopted_prefix_length = 8192;
    assert(server_shared_kv_check_expectation(matching, observation) == server_shared_kv_mismatch::none);

    server_shared_kv_expectation wrong_donor = matching;
    wrong_donor.donor_slot                   = 1;
    assert(server_shared_kv_check_expectation(wrong_donor, observation) == server_shared_kv_mismatch::donor_slot);

    server_shared_kv_expectation wrong_length = matching;
    wrong_length.adopted_prefix_length        = 4096;
    assert(server_shared_kv_check_expectation(wrong_length, observation) ==
           server_shared_kv_mismatch::adopted_prefix_length);
}

static void test_zero_generation_ingest_budget() {
    assert(server_generation_budget_remaining(0, -1, 0) == 0);
    assert(!server_has_generation_budget(0, -1, 0));

    assert(server_generation_budget_remaining(1, -1, 0) == 1);
    assert(server_has_generation_budget(1, -1, 0));
    assert(!server_has_generation_budget(1, -1, 1));

    assert(server_generation_budget_remaining(-1, -1, 0) == -1);
    assert(server_has_generation_budget(-1, -1, 0));
    assert(server_has_generation_budget(-1, -1, 1024));
}

static void test_keeper_context_shift_planning() {
    const std::vector<int>            cached  = { 1, 2, 3, 4, 5, 6 };
    const std::vector<int>            input   = { 1, 2, 5, 6 };
    const server_keeper_context_shift request = {
        /* protected_prefix_tokens */ 2,
        /* expected_cached_tokens  */ cached.size(),
        /* mirror_slot_ids          */ { 0, 1 },
    };

    const auto plan = server_keeper_context_shift_plan_for(request, cached, input);
    assert(plan.mismatch == server_keeper_context_shift_mismatch::none);
    assert(plan.removal_start == 2);
    assert(plan.discard_tokens == 2);

    const std::vector<int> appended      = { 1, 2, 5, 6, 7 };
    const auto             appended_plan = server_keeper_context_shift_plan_for(request, cached, appended);
    assert(appended_plan.mismatch == server_keeper_context_shift_mismatch::none);
    assert(appended_plan.removal_start == 2);
    assert(appended_plan.discard_tokens == 2);

    auto stale = request;
    stale.expected_cached_tokens++;
    assert(server_keeper_context_shift_plan_for(stale, cached, input).mismatch ==
           server_keeper_context_shift_mismatch::cached_length);

    const std::vector<int> substitution = { 1, 2, 9, 6 };
    assert(server_keeper_context_shift_plan_for(request, cached, substitution).mismatch ==
           server_keeper_context_shift_mismatch::substitution);

    const std::vector<int> repeated_cached = { 1, 2, 3, 3, 4 };
    const std::vector<int> repeated_input  = { 1, 2, 3, 4 };
    auto                   repeated        = request;
    repeated.expected_cached_tokens        = repeated_cached.size();
    const auto repeated_plan = server_keeper_context_shift_plan_for(repeated, repeated_cached, repeated_input);
    assert(repeated_plan.mismatch == server_keeper_context_shift_mismatch::none);
    assert(repeated_plan.removal_start == 2);
    assert(repeated_plan.discard_tokens == 1);

    const std::vector<int> ambiguous_cached = { 0, 0, 1, 0 };
    const std::vector<int> ambiguous_input  = { 0, 0, 0 };
    auto                   ambiguous        = request;
    ambiguous.protected_prefix_tokens       = 1;
    ambiguous.expected_cached_tokens        = ambiguous_cached.size();
    assert(server_keeper_context_shift_plan_for(ambiguous, ambiguous_cached, ambiguous_input).mismatch ==
           server_keeper_context_shift_mismatch::ambiguous);

    const std::vector<int> unchanged = cached;
    assert(server_keeper_context_shift_plan_for(request, cached, unchanged).mismatch ==
           server_keeper_context_shift_mismatch::substitution);

    const std::vector<int> too_short = { 1 };
    assert(server_keeper_context_shift_plan_for(request, cached, too_short).mismatch ==
           server_keeper_context_shift_mismatch::bounds);
}

static void test_keeper_shift_preserves_resident_prompt() {
    assert(server_may_replace_resident_prompt_from_host_cache(true, true, true, false));
    assert(!server_may_replace_resident_prompt_from_host_cache(true, true, true, true));
    assert(!server_may_replace_resident_prompt_from_host_cache(false, true, true, false));
    assert(!server_may_replace_resident_prompt_from_host_cache(true, false, true, false));
    assert(!server_may_replace_resident_prompt_from_host_cache(true, true, false, false));
}

int main() {
    test_checkpoint_caps();
    test_checkpoint_retention();
    test_slot_ids();
    test_expectations();
    test_zero_generation_ingest_budget();
    test_keeper_context_shift_planning();
    test_keeper_shift_preserves_resident_prompt();
    return 0;
}
