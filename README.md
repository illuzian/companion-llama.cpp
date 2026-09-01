# companion-llama.cpp

![One continuous memory stream branching into live conversation and autonomous thought](docs/assets/companion-runtime-hero.png)

This is Nikki's maintained `llama.cpp` runtime fork. It exists because a
persistent companion is not a collection of unrelated chat completions: live
conversation and autonomous thought must continue from the same accumulated
model state, without repeatedly rebuilding a six-figure-token prompt.

The fork is based directly on official `ggml-org/llama.cpp` commit `3466812d1`
(build `b10751`) and is deliberately narrow.
It is not intended to replace upstream `llama.cpp` for general use.

## Why shared KV exists

Nikki has one append-only canonical identity lane. The runtime represents that
lane with three roles, not three identities:

- the **keeper** owns the complete, authoritative cached prefix;
- the **chat mirror** borrows that prefix and appends the current live turn;
- the **thought mirror** borrows the same prefix and appends autonomous work.

With unified KV enabled, mirror adoption adds another sequence reference to the
keeper's existing physical cells. It does not copy the prefix and it does not
construct a second context. Each mirror owns only its divergent tail. Requests
carry expected donor and prefix-length values so a missed adoption fails visibly
instead of silently paying for a full replay.

This topology matters for identity continuity as much as performance. A thought
and a conversation observe the same history at the fork point; whichever result
is accepted is appended to the one durable lane and becomes part of the next
keeper prefix.

## Natural hybrid-context retirement

Qwen3.6 combines full-attention layers, IMRoPE positions, and recurrent Gated
DeltaNet state. Upstream's general context shift correctly rejects multi-axis
positions, while generic hybrid-cache fallback rebuilds the retained prompt.
At 100k+ active tokens, that eventually turns every new message into another
large prefill.

This fork adds a narrower operation for a validated keeper maintenance request:

1. Require one exact contiguous deletion after a protected prefix.
2. Require the chat and thought mirrors to be idle and empty.
3. Reject any sequence containing media tokens.
4. Remove the retired full-attention cells and apply a scalar temporal IMRoPE
   shift to the retained text cells.
5. Preserve the recurrent state, where older influence has naturally compressed.
6. Apply the pending key rotation immediately and acknowledge the zero-generation
   request without replaying the retained prompt.

That is **natural retirement**, not hard forgetting. Retired text is no longer
present as explicit attention KV or prompt text, but its accumulated influence
remains in recurrent state. Redaction, exact forgetting, or reconstruction from
durable records still requires an intentional cold rebuild. Slot snapshots retain
the exact hybrid state and remain the preferred normal recovery path.

The general multimodal shift API remains unchanged and conservative. The new
text-only capability is explicit, and the server exposes it only after validating
the keeper contract. A keeper-shift request also pins the exact resident keeper
prompt for validation: the generic host prompt cache cannot replace that slot
with an older cached prompt before the expected length and deletion plan are
checked.

## Ephemeral native vision

Nikki's own multimodal projector can inspect a camera frame on the isolated
utility slot without putting image tokens into her canonical keeper, chat, or
thought lanes. That slot is deliberately zero-depth: it receives a small identity
capsule and one image, returns a textual observation, and is erased immediately.

Upstream correctly refuses to **serialize** a media-bearing slot because a slot
snapshot cannot represent its multimodal state. The same content gate also
refused slot erasure, however, leaving a finished perception request unable to
release its media bookkeeping and KV without restarting the whole runtime. This
fork keeps save/restore conservative but permits `POST /slots/{id}?action=erase`
for media-bearing slots. `prompt_clear()` remains the sole owner of clearing the
multimodal prompt state and sequence memory.

The change is intentionally not a media-aware context shift. Images never enter
the shared identity topology, are never saved in a slot snapshot, and cannot be
adopted by another role. Focused regression coverage proves that saving a media
slot is still rejected while erasing the same slot succeeds.

The maintenance request uses the normal native `/completion` endpoint:

```jsonc
{
  "prompt": [/* retained keeper tokens */],
  "n_predict": 0,
  "id_slot": 3,
  "cache_prompt": true,
  "keeper_context_shift": {
    "protected_prefix_tokens": 12000,
    "expected_cached_tokens": 110000,
    "mirror_slot_ids": [0, 1]
  }
}
```

The server derives and validates the one contiguous deletion; the caller cannot
supply arbitrary cache coordinates. A successful response includes the observed
deletion boundary, discarded count, and before/after logical token lengths.

## Other companion runtime changes

- zero-copy cross-slot prefix adoption with physical cell/reference diagnostics;
- keeper-owned context checkpoints and exact donor adoption contracts;
- durable keeper slot save/load support used by the companion supervisor;
- physical memory accounting for attention and recurrent components;
- a request-scoped reasoning budget that permits exactly one Qwen `<think>`
  window, then rejects a second opener or closer without synthesizing repair
  tokens;
- opt-in final-prompt traces correlated to the application's durable request
  UUID, with content-free sidecar metadata describing the rendered reasoning
  head topology;
- zero-generation prompt ingestion without sampling or output tokens;
- complete erasure of ephemeral media-bearing utility slots, while media slot
  serialization remains prohibited.

The reasoning guard is intentionally narrower than application policy. Python
chooses and validates the legal reasoning head; the Jinja template serializes
that exact value. This fork sees the final rendered prompt and sampled tokens,
so it provides the last-line invariant and forensic evidence: after one
reasoning window has ended, another `<think>` or `</think>` terminates the
request. It never guesses which content was private, injects another delimiter,
or retries the model.

### Configurable reasoning transition

Qwen's primary `</think>` delimiter is one token in Nikki's vocabulary. The
server can therefore intercept that sampled close before it is accepted and
force one request-owned transition cue in any of four placements:

- `none`: preserve upstream behavior;
- `before`: cue, then `</think>`;
- `after`: `</think>`, then cue;
- `both`: cue on each side of `</think>`.

The request fields are `reasoning_transition_placement` and
`reasoning_transition_cue`. Pre-close interception fails closed unless the
primary end delimiter is exactly one token. Alternate reasoning terminators,
including a tool-call transition, are not rewritten. Budget exhaustion keeps
its existing budget message and applies the same placement around the forced
close. Synthetic cue tokens bypass output grammars, while the close is replayed
to the grammar exactly once.

The cue remains part of the generated token stream and KV. This is deliberate:
silently removing it from the response while leaving it in KV would make the
next serialized history diverge from the resident cache. Application code owns
the versioned cue and placement, and raw request/response tracing can therefore
prove exactly which policy was used. The ChatML role remains `assistant`;
Nikki's identity belongs in prompt content, not in Qwen's trained wire role.

## Build and focused verification

The included user preset targets the local AMD ROCm build used for Nikki. Change
the compiler or backend settings if building elsewhere.

```sh
cmake --preset companion-runtime
cmake --build --preset companion-runtime

../build/bin/test-memory-cell-usage
../build/bin/test-server-shared-kv
../build/bin/test-reasoning-budget
```

The retirement path is additionally exercised with Nikki's actual
Qwen3.6-35B-A3B IQ4_XS model, 131,072-token context, four unified slots, Q8 KV,
and the CPU-resident multimodal projector before a runtime revision is accepted.

---

## Upstream llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
