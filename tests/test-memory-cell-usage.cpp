#include "../src/llama-batch.h"
#include "../src/llama-kv-cells.h"
#include "../src/llama-memory.h"
#include "llama.h"

#undef NDEBUG
#include <cassert>
#include <map>
#include <vector>

struct diagnostic_memory final : llama_memory_i {
    llama_kv_cells cells;

    llama_memory_context_ptr init_batch(llama_batch_allocr &, uint32_t, bool) override { return nullptr; }

    llama_memory_context_ptr init_full() override { return nullptr; }

    llama_memory_context_ptr init_update(llama_context *, bool) override { return nullptr; }

    bool get_can_shift() const override { return true; }

    void clear(bool) override { cells.reset(); }

    bool seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) override {
        for (uint32_t index = 0; index < cells.size(); ++index) {
            if (cells.is_empty(index) || !cells.pos_in(index, p0, p1) || !cells.seq_has(index, seq_id)) {
                continue;
            }
            cells.seq_rm(index, seq_id);
        }
        return true;
    }

    void seq_cp(llama_seq_id source, llama_seq_id destination, llama_pos p0, llama_pos p1) override {
        for (uint32_t index = 0; index < cells.size(); ++index) {
            if (cells.is_empty(index) || !cells.pos_in(index, p0, p1) || !cells.seq_has(index, source) ||
                cells.seq_has(index, destination)) {
                continue;
            }
            cells.seq_add(index, destination);
        }
    }

    void seq_keep(llama_seq_id) override { assert(false); }

    void seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override {
        for (uint32_t index = 0; index < cells.size(); ++index) {
            if (cells.is_empty(index) || !cells.pos_in(index, p0, p1) || !cells.seq_has(index, seq_id)) {
                continue;
            }
            cells.pos_add(index, shift);
        }
    }

    void seq_div(llama_seq_id, llama_pos, llama_pos, int) override { assert(false); }

    llama_pos seq_pos_min(llama_seq_id seq_id) const override { return cells.seq_pos_min(seq_id); }

    llama_pos seq_pos_max(llama_seq_id seq_id) const override { return cells.seq_pos_max(seq_id); }

    std::vector<llama_memory_cell_usage> get_cell_usage(llama_seq_id seq_id) const override {
        llama_memory_cell_usage usage = cells.get_cell_usage(seq_id);
        usage.type                    = LLAMA_MEMORY_CELL_TYPE_ATTENTION;
        return { usage };
    }

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override { return {}; }

    void state_write(llama_io_write_i &, llama_seq_id, llama_state_seq_flags) const override { assert(false); }

    void state_read(llama_io_read_i &, llama_seq_id, llama_state_seq_flags) override { assert(false); }
};

static llama_memory_cell_usage usage(diagnostic_memory & memory, llama_seq_id sequence_id) {
    assert(llama_memory_get_cell_usage(&memory, sequence_id, nullptr, 0) == 1);

    llama_memory_cell_usage result = {};
    assert(llama_memory_get_cell_usage(&memory, sequence_id, &result, 1) == 1);
    assert(result.component_index == 0);
    assert(result.type == LLAMA_MEMORY_CELL_TYPE_ATTENTION);
    assert(result.sequence_cells == result.sequence_shared_cells + result.sequence_exclusive_cells);
    return result;
}

static void seed_sequence(diagnostic_memory & memory, llama_seq_id sequence_id, uint32_t first_cell, uint32_t count) {
    for (uint32_t offset = 0; offset < count; ++offset) {
        const uint32_t index = first_cell + offset;
        memory.cells.pos_set(index, offset);
        memory.cells.seq_add(index, sequence_id);
    }
}

int main() {
    constexpr llama_seq_id chat          = 0;
    constexpr llama_seq_id thought       = 1;
    constexpr llama_seq_id utility       = 2;
    constexpr llama_seq_id keeper        = 3;
    constexpr uint32_t     prefix_length = 4;

    diagnostic_memory memory;
    memory.cells.resize(8);
    seed_sequence(memory, keeper, 0, prefix_length);

    llama_memory_cell_usage current = usage(memory, keeper);
    assert(current.capacity_cells == 8);
    assert(current.occupied_cells == prefix_length);
    assert(current.sequence_references == prefix_length);
    assert(current.duplicate_sequence_references == 0);
    assert(current.shared_cells == 0);
    assert(current.sequence_cells == prefix_length);
    assert(current.sequence_exclusive_cells == prefix_length);

    // Unified-memory adoption adds sequence references to the keeper's cells.
    // No physical cell is copied or separately reserved for either mirror.
    memory.seq_cp(keeper, chat, 0, prefix_length);
    memory.seq_cp(keeper, thought, 0, prefix_length);
    current = usage(memory, chat);
    assert(current.occupied_cells == prefix_length);
    assert(current.sequence_references == 3 * prefix_length);
    assert(current.duplicate_sequence_references == 2 * prefix_length);
    assert(current.shared_cells == prefix_length);
    assert(current.sequence_cells == prefix_length);
    assert(current.sequence_shared_cells == prefix_length);
    assert(current.sequence_exclusive_cells == 0);

    // Utility cells occupy independent storage even when their positions overlap.
    seed_sequence(memory, utility, prefix_length, 2);
    current = usage(memory, utility);
    assert(current.occupied_cells == prefix_length + 2);
    assert(current.sequence_references == 3 * prefix_length + 2);
    assert(current.duplicate_sequence_references == 2 * prefix_length);
    assert(current.sequence_cells == 2);
    assert(current.sequence_shared_cells == 0);
    assert(current.sequence_exclusive_cells == 2);

    // Reset drops only the mirror references. Re-adoption returns to the same
    // physical occupancy and recreates shared references without copied cells.
    memory.seq_rm(chat, 0, prefix_length);
    current = usage(memory, chat);
    assert(current.occupied_cells == prefix_length + 2);
    assert(current.sequence_references == 2 * prefix_length + 2);
    assert(current.duplicate_sequence_references == prefix_length);
    assert(current.sequence_cells == 0);

    memory.seq_cp(keeper, chat, 0, prefix_length);
    current = usage(memory, chat);
    assert(current.occupied_cells == prefix_length + 2);
    assert(current.sequence_references == 3 * prefix_length + 2);
    assert(current.duplicate_sequence_references == 2 * prefix_length);
    assert(current.sequence_shared_cells == prefix_length);

    // Removing and re-seeding the keeper reference likewise preserves the
    // mirror-owned physical prefix, proving symmetric re-adoption semantics.
    memory.seq_rm(keeper, 0, prefix_length);
    current = usage(memory, keeper);
    assert(current.occupied_cells == prefix_length + 2);
    assert(current.sequence_references == 2 * prefix_length + 2);
    assert(current.sequence_cells == 0);

    memory.seq_cp(chat, keeper, 0, prefix_length);
    current = usage(memory, keeper);
    assert(current.occupied_cells == prefix_length + 2);
    assert(current.sequence_references == 3 * prefix_length + 2);
    assert(current.sequence_shared_cells == prefix_length);

    // A keeper-only middle retirement preserves the pinned head, frees the
    // discarded cells, and shifts only the retained keeper suffix.
    diagnostic_memory shifted;
    shifted.cells.resize(8);
    seed_sequence(shifted, keeper, 0, 6);
    shifted.seq_cp(keeper, chat, 0, 6);
    shifted.seq_cp(keeper, thought, 0, 6);
    shifted.seq_rm(chat, 0, 6);
    shifted.seq_rm(thought, 0, 6);

    shifted.seq_rm(keeper, 2, 4);
    shifted.seq_add(keeper, 4, 6, -2);
    current = usage(shifted, keeper);
    assert(current.occupied_cells == 4);
    assert(current.sequence_cells == 4);
    assert(current.sequence_shared_cells == 0);
    assert(shifted.cells.pos_get(0) == 0);
    assert(shifted.cells.pos_get(1) == 1);
    assert(shifted.cells.is_empty(2));
    assert(shifted.cells.is_empty(3));
    assert(shifted.cells.pos_get(4) == 2);
    assert(shifted.cells.pos_get(5) == 3);

    return 0;
}
