# Natural language understanding
Credits: much of this task was developed by Mirella Lapata, Frank Keller, Clara Vania and Adam Lopez in School of Informatics, University of Edinburgh. And Richard Socher's cs224d course in Standford.

### Predicting Subject-Verb Agreement
The form of an English third-person present tense verb depends on whether the head of the syntactic subject is plural or singular. For example, native English speakers strongly prefer sentences (i) and (iv) below, and regard (ii) and (iii) as ungrammatical:
i) The key is on the table.
ii) *The key are on the table.
iii) *The keys is on the table.
iv) The keys are on the table.

This agreement tends to persist even when the head of the subject is separated from the verb by intervening words:
v) The keys to the cabinet are on the table.

Agreement rules like this are widespread, and are often more complex than in English. Our goal for this question will be to test (in a limited way) whether an RNN can also learn them. For our first test, we will train a model predict agreement using direct supervision. That is, we will give our model the sequence of words preceding the verb, and we will ask it to predict whether the verb is singular (VBZ), or plural (VBP). Our training and test data will be in this form:

    Input: The keys to the cabinet
    Output: VBP

Since the head of the subject may be arbitrarily far from the verb, this problem is a natural application of RNNs, which can encode the input sentence. But since the task is now binary classification, we must make some changes to the RNN. Instead of making predictions at every time step, we only make a prediction at the final time step.

### Number Prediction with an RRNLM
Consider the problem from [Predicting Subject-Verb Agreement](#predicting-subject-verb-agreement) from a slightly different perspective, inspired by the Linzen et al. (2016) paper that provided our data. Human learners of language generally learn morphosyntactic features of language, like number agreement, with little to no direct supervision. Can a computational model like an RNN also do this? That is, can it learn agreement simply from the language data itself?

    Input x = The keys to the cabinet
    Output = VBZ if P(is | x) > P(are | x) else VBP

### References
* Jiang Guo. Backpropagation Through Time. Unpubl. ms., Harbin Institute of Technology, 2013.
* Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. Assessing the ability of LSTMs to learn syntax-sensitive dependencies. Transactions of the Association for Compu-
tational Linguistics, 4:521–535, 2016.
* Tomas Mikolov, Martin Karafiát, Lukas Burget, Jan Cernock` y, and Sanjeev Khudanpur. Recurrent neural network based language model. In INTERSPEECH, volume 2, page 3, 2010.
