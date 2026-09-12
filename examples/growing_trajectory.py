"""Infrastructure demonstration only: replace these turns with your task."""

from llm_behavior_pipeline import GenerationConfig, Trajectory, make_llm


model = make_llm("mock")
trajectory: Trajectory = []
config = GenerationConfig(temperature=0.0, max_tokens=50, seed=123)

for trial_text in ["This is trial one.", "This is trial two."]:
    trajectory.append({"role": "user", "content": trial_text})
    result = model.respond_with_metadata(trajectory, config=config)
    trajectory.append({"role": "assistant", "content": result.text})
    print(result.text, result.response)
