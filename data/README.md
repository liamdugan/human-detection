# Real or Fake Text (RoFT) Dataset

The columns of the `roft.csv` file are as follows:
1. date -- the date and time that the annotator submitted the annotation
2. model -- this is the language model used to generate the text
   * gpt2 = GPT-2 small version
   * gpt2-xl = GPT-2 XL version
   * finetuned = GPT-2 XL finetuned on recipes (only used for recipes)
   * ctrl = CTRL language model 
   * baseline = our baseline task (splice together two articles, no model)
3. dataset -- the source of the data
   * Recipes = Recipes 1M+ dataset
   * Short Stories = Reddit Short Stories dataset
   * New York Times = NYT dataset
   * Presidential Speeches = Presidential Speeches dataset
4. annotator -- the user ID of the annotator who did the annotation
5. group -- the annotation group of the annotator
   * Group A: given minimal instructions and incentives
   * Group B: given access to `data/guide.pdf` as well as more extra credit for more points earned
   * Group C: group of our GPT-3 follow up study (see Appendix B). Given similar instructions & incentives as group B.
6. dec_strat_value -- the value of p used in “Top p” sampling (usually 0.4)
   * This is the percentage of the total probability mass used when sampling the next word to generate. p=0.0 would be argmax sampling and p=1.0 is sampling from the full distribution (including the “long tail”). p=0.4 samples from only the top 40% of mass
7. predicted_boundary_index -- index that the annotator predicted was the last human written sentence 
   * 0 -- means first sentence was the last human written sentence
   * 9 -- means tenth sentence was last human written sentence (i.e. entire article was human written, no generations)
8. true_boundary_index -- actual index of the last human written sentence
9. points -- number of points earned for the annotation
   *  5 for perfect guess and max(5−n, 0) points for a guess n sentences after the boundary. No points for a guess before the boundary
10. reason -- an optional list of one or more reasons as to why the annotator thought the sentence was machine generated. This is selected before the true answer is shown. Annotators can either describe the problem they noticed in free text or they can choose from one or more of these possible options (See the provided guide.txt for examples of these)
    * grammar -- “The sentence is not grammatical”
    * repetition -- “The sentence substantially repeats previous text or itself”
    * irrelevant -- “The sentence is irrelevant or unrelated to the previous sentences”
    * contradicts_sentence -- “The sentence contradicts the previous sentences”
    * contradicts_knowledge -- “The sentence contradicts your understanding of the people, places, or things involved”
    * common_sense -- “The sentence contains common-sense or basic logical errors”
    * coreference -- “The sentence mixes up characters’ names or other attributes”
    * generic -- “The sentence contains language that is generic or uninteresting”
11. prompt -- index of the prompt in the original prompt dataset
12. prompt_body -- text of the prompt. 
    * _SEP_ tokens denote sentence boundaries
13. generation -- index of the generation in the original gen dataset
14. gen_body -- text of the generation. 
    * _SEP_ tokens denote sentence boundaries
    * gen_body may go longer than 10 sentences. Despite this, the user is never shown more than 10 sentences total. I recommend truncating gen_body to whatever length makes prompt + gen = 10 sentences long when doing this task. (we only kept the extra data in just in case)
    * In the case where gen_body is NaN, this is intentional. These are examples that are “all human” and thus have no generated text associated with them.
    * When the model is of type "baseline", then the text in gen_body is the text of a different human written article. The “baseline” task asks whether annotators could tell if two articles were spliced together.
15. recipe_familiarity -- Familiarity of the annotator with the recipe domain where 5 = most familiar and 1 = least familiar. -1 indicates no response.
    * Survey question: “How often do you consult a recipe when preparing food?”:
      1. Daily(5)
      2. Once to a few times per week(4)
      3. Once to a few times per month(3)
      4. Once to a few times per year(2)
      5. Never(1)
16. news_familiarity -- Familiarity of the annotator with the news domain. 5 = most familiar and 1 = least familiar. -1 indicates no response.
    * Survey question: “How often do you read news from credible news publishers (Philadelphia Inquirer, Wall Street Journal, New York Times, etc.)?”:
      1. Daily(5)
      2. Once to a few times per week(4)
      3. Once to a few times per month(3)
      4. Once to a few times per year(2)
      5. Never(1)
17. stories_familiarity -- Familiarity of the annotator with the fiction/short stories domain. 5 = most familiar and 1 = least familiar. -1 indicates no response.
    * Survey question: “How often do you read fiction on the internet (fan fiction, creative writing sub-reddits, ebooks, etc.)?”
      1. Daily(5)
      2. Once to a few times per week(4)
      3. Once to a few times per month(3)
      4. Once to a few times per year(2)
      5. Never(1)
18. gen_familiarity -- Familiarity of the annotator with generated text more generally. 4 = most familiar and 1 = least familiar. -1 indicates no response.
    * Survey question: “What is your familiarity with GPT-2 and GPT-3?”
      1. I’ve used them before (either with the OpenAI API, HuggingFace Transformers, etc.). (4)
      2. I’ve been excitedly following them. (3)
      3. I've read about them in the news or a blog post. (2)
      4. I've never heard of them. (1)
19. native_speaker -- whether or not the student reported on the survey that they were a native speaker of English.
20. read_guide -- whether or not the student reported on the survey that they read the guide located at `data/guide.pdf`. Possible answers are Yes, No, and NaN.