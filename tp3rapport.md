There are actually two common approaches to padding attention masks:

Bidirectional masking (like in BERT)Full padding masking
What you're observing is the bidirectional masking approach. This is actually common and often intentional because:

It's simpler to implement
It doesn't affect the final output since pad token representations are typically ignored in subsequent layers
In encoder architectures (like BERT), allowing padding tokens to attend doesn't harm the model since their representations aren't used for predictions