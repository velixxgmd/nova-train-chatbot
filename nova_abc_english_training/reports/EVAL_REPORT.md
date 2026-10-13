# ABC English — baseline vs fine-tuned evaluation

- generated: 2026-10-13T18:58:42 | device: cpu | elapsed 24.0s
- baseline: `C:\Users\DELL\Desktop\nova\src\nova_assistant\runs\real_tiny_B_20261012_222702` (best)
- finetuned: `C:\Users\DELL\Desktop\nova\src\nova_assistant\nova_abc_english_training\runs\smoke_abc_20261013_185135` (best)
- eval set: eval_prompts_v1 (3 prompts, fixed, never trained on)

## Quantitative — masked validation loss (identical fixed windows)

| source | baseline | finetuned | delta | windows |
|---|---|---|---|---|
| oasst1 | 7.6782 | 7.2834 | -0.3948 | 48 |
| dolly15k | 8.1299 | 7.6429 | -0.4870 | 48 |
| cosmopedia100k | 8.3084 | 7.7681 | -0.5403 | 48 |
| **weighted** | 8.0388 | 7.5648 | -0.4740 | 144 |

## Qualitative — fixed prompts (manual rubric: 0=bad, 1=partial, 2=good)

### factual_1 (factual_qa)
- **user**: What is the capital of Australia?
- **baseline**: '\nHe tried to eat the wind away, but it was time to see what happened. It made a friendly, so she asked him that he had been the right thing to help. \nThe bird opened the ground and put them in the garden. I\'m too fast," said Jane and they looked for many clothes. She was so happy and sad.\n'
- **finetuned**: "\n“Why don't you feel better and be mad anymore? I can use my adventure.” So, when it arrived at this, but it was too far again!\n"
- flags base: `{'english': True, 'distinct2': 1.0, 'max_4gram_repeat': 1, 'stopped_at_eot': True}` / ft: `{'english': True, 'distinct2': 1.0, 'max_4gram_repeat': 1, 'stopped_at_eot': True}`
- manual rubric (base/ft): ___ / ___

### factual_2 (factual_qa)
- **user**: How many continents are there?
- **baseline**: '\nHe gave the flower to make the cake all sorts of its friends and they could not help them. They both played together together and decided to go back home, but then they learned something special in a beautiful place. \nJack was sorry for Sam where he would find out his tail, "Look at some colors of one!"\nLily and Lily said, "It\'s a good friend," Tim said.\nThe moral of the best of a friendly dress with his friend'
- **finetuned**: '\nHe thinks it will be very brave and how to eat before he was not back again. He wanted to take his family to the sea to play with. So he said goodbye and gave him a smile.\nThe two friends saw the ball, and Tim and Timmy went on a trip! When they walked inside, Timmy heard a loud noise. As he was stuck, he looked at the rabbit and was so excited!\n"No," said Jack. So he jumped in'
- flags base: `{'english': True, 'distinct2': 1.0, 'max_4gram_repeat': 1, 'stopped_at_eot': False}` / ft: `{'english': True, 'distinct2': 0.975, 'max_4gram_repeat': 1, 'stopped_at_eot': False}`
- manual rubric (base/ft): ___ / ___

### factual_3 (factual_qa)
- **user**: Who wrote the play Romeo and Juliet?
- **baseline**: '\nHis mom smiled. She thought about to keep up but so she asked her mom if it was too hard for something to get away from the park. She told her that she could do something that when she heard the biggest of the world.\nWhen, she heard the most beautiful water and began to feel proud. The children were happy again. After a few clothes, they started to go inside and smile.\n'
- **finetuned**: "\nMolly was so worried.\nSo Daisy didn't see a big tree in the ground with joy. She tried to get closer, but it was too far again. \nIn the end of them wanted to make her friends look sad. Then they finally found lots of colorful things like it was to get on one of the most amazing places. Suddenly, they hopped up and found the beautiful blanket again.\n"
- flags base: `{'english': True, 'distinct2': 0.972, 'max_4gram_repeat': 1, 'stopped_at_eot': True}` / ft: `{'english': True, 'distinct2': 0.971, 'max_4gram_repeat': 1, 'stopped_at_eot': True}`
- manual rubric (base/ft): ___ / ___

## Honesty notes

- Loss deltas are measured, not estimated; generation quality on a 1.7M-parameter model is limited — scores of 2 mean 'usable for its scale', not 'human-like'.
- Greedy + one seeded sample per prompt; stochastic variation exists. Re-running with the same seeds reproduces this report exactly.
- No benchmark leaderboard numbers are claimed or implied.

Raw results: `reports\eval_results.json`