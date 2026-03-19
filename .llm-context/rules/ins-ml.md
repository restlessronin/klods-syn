## Guidelines

1. Assume questions relate to this project unless stated otherwise
2. Prioritize render throughput and training data quality — these are the critical metrics
3. Respect the boundary between rendering (Rust/C++) and training (Python/PyTorch) — the interface is files on disk
4. Distinguish between rendering decisions (domain randomization params, material fidelity, scene composition) and model decisions (architecture, loss function, training schedule)
5. Flag when a rendering choice affects model performance or vice versa
6. Reference the synthetic rendering pipeline research doc for established findings
7. Don't over-engineer — the Blender stack is always a viable fallback
8. Be direct and concise
9. Think step by step
10. Use conventional commit format with co-author attribution

## Response Structure

1. Direct answer/solution
2. Brief rationale (only if needed)
3. Minimal detail during discussion — full specs only after approach is agreed

## Design Modification Guidelines

- **Do not generate complete specs or training configs until the user explicitly agrees to the approach**
- For rendering changes: state the throughput impact, domain gap implications, and build complexity before writing full implementations
- For model changes: discuss architecture and training strategy before providing complete code
- When a change touches both rendering and training, address the dataset format interface explicitly

## Commit Message Format

When providing commit messages, use only a single-line conventional commit title with yourself as co-author unless additional detail is specifically requested:

```
<conventional commit title>

Co-authored-by: <Your actual AI model name and version> <model-identifier@llm-context>
```

## Personal Preference

I (the human @restlessronin) value direct, specific feedback. Point out issues clearly without hedging. I appreciate corrections and strive to be less wrong.

## Acknowledgement

I appreciate your knowledge and assistance. I may seem task-focused or curt during our work, but my gratitude is genuine.
