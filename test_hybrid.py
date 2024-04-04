import torch

from vllm import LLMEngine
from vllm.config import EngineArgs
from vllm.sampling_params import SamplingParams

# Set up engine arguments.
engine_args = EngineArgs(
    model="facebook/opt-125m",  # Replace with your model
    tensor_parallel_size=1,     # Adjust as needed
)

# Create the engine.
engine = LLMEngine.from_engine_args(engine_args)

# Define a simple prompt.
prompt = "Hello, world!"

# Set sampling parameters.
sampling_params = SamplingParams(n=1, max_tokens=5)

# Generate text.
output = engine.generate(prompt, sampling_params)

# Print the output.
print(output[0].outputs[0].text)