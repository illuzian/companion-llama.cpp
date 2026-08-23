#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

struct server_shared_kv_expectation {
    std::optional<int32_t> donor_slot;
    std::optional<size_t>  adopted_prefix_length;
};

struct server_shared_kv_observation {
    int32_t donor_slot            = -1;
    size_t  adopted_prefix_length = 0;
};

struct server_keeper_context_shift {
    size_t               protected_prefix_tokens = 0;
    size_t               expected_cached_tokens  = 0;
    std::vector<int32_t> mirror_slot_ids;
};

struct server_keeper_context_shift_observation {
    bool   applied               = false;
    size_t removal_start         = 0;
    size_t discarded_tokens      = 0;
    size_t logical_tokens_before = 0;
    size_t logical_tokens_after  = 0;
};

enum class server_keeper_context_shift_mismatch {
    none,
    cached_length,
    bounds,
    substitution,
    ambiguous,
};

struct server_keeper_context_shift_plan {
    server_keeper_context_shift_mismatch mismatch       = server_keeper_context_shift_mismatch::none;
    size_t                               removal_start  = 0;
    size_t                               discard_tokens = 0;
};

inline bool server_may_replace_resident_prompt_from_host_cache(bool update_requested,
                                                               bool prompt_cache_available,
                                                               bool completion_task,
                                                               bool keeper_context_shift_requested) {
    // A keeper shift is validated against the exact resident prompt named by
    // expected_cached_tokens. Replacing that prompt from the host cache before
    // validation can swap in an older logical context and manufacture a stale
    // cached-length mismatch.
    return update_requested && prompt_cache_available && completion_task && !keeper_context_shift_requested;
}

template <typename CachedTokens, typename InputTokens>
server_keeper_context_shift_plan server_keeper_context_shift_plan_for(const server_keeper_context_shift & shift,
                                                                      const CachedTokens &                cached_tokens,
                                                                      const InputTokens & input_tokens) {
    if (cached_tokens.size() != shift.expected_cached_tokens) {
        return { server_keeper_context_shift_mismatch::cached_length, 0, 0 };
    }

    if (shift.protected_prefix_tokens == 0 || shift.protected_prefix_tokens >= cached_tokens.size() ||
        input_tokens.size() < shift.protected_prefix_tokens) {
        return { server_keeper_context_shift_mismatch::bounds, 0, 0 };
    }

    size_t common_prefix = 0;
    while (common_prefix < cached_tokens.size() && common_prefix < input_tokens.size() &&
           cached_tokens[common_prefix] == input_tokens[common_prefix]) {
        ++common_prefix;
    }
    if (common_prefix < shift.protected_prefix_tokens) {
        return { server_keeper_context_shift_mismatch::substitution, 0, 0 };
    }

    // Match prefixes of the reversed cached prompt against substrings ending at
    // each boundary in the reversed input. A match of length N at boundary S
    // proves cached[cached.size() - N:] == input[S:S + N].
    const size_t        cached_size = cached_tokens.size();
    const size_t        input_size  = input_tokens.size();
    std::vector<size_t> failure(cached_size, 0);
    for (size_t index = 1; index < cached_size; ++index) {
        size_t       matched = failure[index - 1];
        const auto & token   = cached_tokens[cached_size - 1 - index];
        while (matched > 0 && token != cached_tokens[cached_size - 1 - matched]) {
            matched = failure[matched - 1];
        }
        if (token == cached_tokens[cached_size - 1 - matched]) {
            ++matched;
        }
        failure[index] = matched;
    }

    std::vector<size_t> suffix_matches(input_size + 1, 0);
    size_t              matched = 0;
    for (size_t consumed = 1; consumed <= input_size; ++consumed) {
        if (matched == cached_size) {
            matched = failure[matched - 1];
        }
        const auto & token = input_tokens[input_size - consumed];
        while (matched > 0 && token != cached_tokens[cached_size - 1 - matched]) {
            matched = failure[matched - 1];
        }
        if (token == cached_tokens[cached_size - 1 - matched]) {
            ++matched;
        }
        suffix_matches[consumed] = matched;
    }

    bool         found        = false;
    bool         ambiguous    = false;
    size_t       best_suffix  = 0;
    size_t       best_start   = 0;
    size_t       best_end     = 0;
    const size_t latest_start = std::min(common_prefix, cached_size - 1);
    for (size_t start = shift.protected_prefix_tokens; start <= latest_start; ++start) {
        size_t suffix = suffix_matches[input_size - start];
        while (suffix >= cached_size - start) {
            suffix = suffix == 0 ? 0 : failure[suffix - 1];
        }

        // A zero-length suffix cannot distinguish retirement from arbitrary
        // replacement of the entire rolling body, so fail closed.
        if (suffix == 0) {
            continue;
        }

        const size_t end = cached_size - suffix;
        if (!found || suffix > best_suffix) {
            found       = true;
            ambiguous   = false;
            best_suffix = suffix;
            best_start  = start;
            best_end    = end;
        } else if (suffix == best_suffix && (start != best_start || end != best_end)) {
            ambiguous = true;
        }
    }

    if (!found) {
        return { server_keeper_context_shift_mismatch::substitution, 0, 0 };
    }
    if (ambiguous) {
        return { server_keeper_context_shift_mismatch::ambiguous, 0, 0 };
    }

    return { server_keeper_context_shift_mismatch::none, best_start, best_end - best_start };
}

enum class server_shared_kv_mismatch {
    none,
    donor_slot,
    adopted_prefix_length,
};

inline server_shared_kv_mismatch server_shared_kv_check_expectation(const server_shared_kv_expectation & expectation,
                                                                    const server_shared_kv_observation & observation) {
    if (expectation.donor_slot && *expectation.donor_slot != observation.donor_slot) {
        return server_shared_kv_mismatch::donor_slot;
    }

    if (expectation.adopted_prefix_length && *expectation.adopted_prefix_length != observation.adopted_prefix_length) {
        return server_shared_kv_mismatch::adopted_prefix_length;
    }

    return server_shared_kv_mismatch::none;
}

inline bool server_slot_id_is_valid(int32_t id_slot, size_t n_slots, bool allow_automatic) {
    if (allow_automatic && id_slot == -1) {
        return true;
    }

    return id_slot >= 0 && static_cast<size_t>(id_slot) < n_slots;
}

inline int32_t server_generation_budget_remaining(int32_t request_n_predict,
                                                  int32_t global_n_predict,
                                                  int32_t n_decoded) {
    if (request_n_predict == -1 && global_n_predict == -1) {
        return -1;
    }

    const int32_t limit = request_n_predict != -1 ? request_n_predict : global_n_predict;
    return limit - n_decoded;
}

inline bool server_has_generation_budget(int32_t request_n_predict, int32_t global_n_predict, int32_t n_decoded) {
    const int32_t remaining = server_generation_budget_remaining(request_n_predict, global_n_predict, n_decoded);
    return remaining == -1 || remaining > 0;
}

inline int32_t server_effective_checkpoint_cap(int32_t configured_cap, int32_t keeper_slot, int32_t slot_id) {
    const int32_t nonnegative_cap = std::max<int32_t>(0, configured_cap);

    if (keeper_slot < 0 || slot_id == keeper_slot) {
        return nonnegative_cap;
    }

    return 0;
}

template <typename CheckpointContainer>
inline void server_apply_checkpoint_cap(CheckpointContainer & checkpoints, int32_t checkpoint_cap) {
    if (checkpoint_cap <= 0) {
        checkpoints.clear();
        return;
    }

    while (checkpoints.size() > static_cast<size_t>(checkpoint_cap)) {
        checkpoints.erase(checkpoints.begin());
    }
}
