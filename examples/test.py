import dspy

lm = dspy.LM('bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0')
dspy.configure(lm=lm)

print(lm("What is 2+2?", temperature=0.9))
