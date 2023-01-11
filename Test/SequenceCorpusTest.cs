using System.Collections.Generic;
using Classification.Parameter;
using Dictionary.Dictionary;
using NUnit.Framework;
using SequenceProcessing.Classification;
using SequenceProcessing.Sequence;
using WordToVec;

namespace Test
{
    public class SequenceCorpusTest
    {
        
        private void VectorizedCorpus(SequenceCorpus corpus, VectorizedDictionary dictionary) {
            for (var i = 0; i < corpus.SentenceCount(); i++) {
                var sentence = corpus.GetSentence(i);
                for (var j = 0; j < sentence.WordCount(); j++) {
                    var word = (LabelledVectorizedWord) sentence.GetWord(j);
                    var vectorizedWord = (VectorizedWord) dictionary.GetWord(word.GetName());
                    sentence.ReplaceWord(j, new LabelledVectorizedWord(word.GetName(), vectorizedWord.GetVector(), word.GetClassLabel()));
                }
            }
        }
        
        [Test]
        public void TestRNN() {
            var corpus = new SequenceCorpus("postag-atis-tr.txt");
            var testCorpus = new SequenceCorpus("postag-atis-tr-test.txt");
            var neuralNetwork = new NeuralNetwork(corpus, new WordToVecParameter());
            var dictionary = neuralNetwork.Train();
            VectorizedCorpus(corpus, dictionary);
            VectorizedCorpus(testCorpus, dictionary);
            int correct = 0, total = 0;
            var hidden = new List<int>();
            hidden.Add(10);
            var model = new RecurrentNeuralNetworkModel(corpus, new DeepNetworkParameter(1, 0.01, 0.99, 0.9, 100, hidden, ActivationFunction.SIGMOID));
            for (var i = 0; i < testCorpus.SentenceCount(); i++) {
                var sentence = testCorpus.GetSentence(i);
                var list = model.Predict(sentence);
                for (var j = 0; j < list.Count; j++) {
                    var word = (LabelledVectorizedWord) sentence.GetWord(j);
                    if (list[j].Equals(word.GetClassLabel())) {
                        correct++;
                    }
                    total++;
                }
            }
            Assert.AreEqual(97.79595765158807, (correct * 100.00) / (total + 0.00));
        }
        
        [Test]
        public void TestCorpus01()
        {
            var corpus = new SequenceCorpus("../../../disambiguation-penn.txt");
            Assert.AreEqual(25957, corpus.SentenceCount());
            Assert.AreEqual(264930, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus02()
        {
            var corpus = new SequenceCorpus("../../../postag-atis-en.txt");
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(61879, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus03()
        {
            var corpus = new SequenceCorpus("../../../slot-atis-en.txt");
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(61879, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus04()
        {
            var corpus = new SequenceCorpus("../../../slot-atis-tr.txt");
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(45875, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus05()
        {
            var corpus = new SequenceCorpus("../../../disambiguation-atis.txt");
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(45875, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus06()
        {
            var corpus = new SequenceCorpus("../../../metamorpheme-atis.txt");
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(45875, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus07()
        {
            var corpus = new SequenceCorpus("../../../postag-atis-tr.txt");
            Assert.AreEqual(5432, corpus.SentenceCount());
            Assert.AreEqual(45875, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus08()
        {
            var corpus = new SequenceCorpus("../../../metamorpheme-penn.txt");
            Assert.AreEqual(25957, corpus.SentenceCount());
            Assert.AreEqual(264930, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus09()
        {
            var corpus = new SequenceCorpus("../../../ner-penn.txt");
            Assert.AreEqual(19118, corpus.SentenceCount());
            Assert.AreEqual(168654, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus10()
        {
            var corpus = new SequenceCorpus("../../../postag-penn.txt");
            Assert.AreEqual(25957, corpus.SentenceCount());
            Assert.AreEqual(264930, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus11()
        {
            var corpus = new SequenceCorpus("../../../semanticrolelabeling-penn.txt");
            Assert.AreEqual(19118, corpus.SentenceCount());
            Assert.AreEqual(168654, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus12()
        {
            var corpus = new SequenceCorpus("../../../semantics-penn.txt");
            Assert.AreEqual(19118, corpus.SentenceCount());
            Assert.AreEqual(168654, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus13()
        {
            var corpus = new SequenceCorpus("../../../shallowparse-penn.txt");
            Assert.AreEqual(9557, corpus.SentenceCount());
            Assert.AreEqual(87279, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus14()
        {
            var corpus = new SequenceCorpus("../../../disambiguation-tourism.txt");
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus15()
        {
            var corpus = new SequenceCorpus("../../../metamorpheme-tourism.txt");
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus16()
        {
            var corpus = new SequenceCorpus("../../../postag-tourism.txt");
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus17()
        {
            var corpus = new SequenceCorpus("../../../semantics-tourism.txt");
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus18()
        {
            var corpus = new SequenceCorpus("../../../shallowparse-tourism.txt");
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus19()
        {
            var corpus = new SequenceCorpus("../../../disambiguation-kenet.txt");
            Assert.AreEqual(18687, corpus.SentenceCount());
            Assert.AreEqual(178658, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus20()
        {
            var corpus = new SequenceCorpus("../../../metamorpheme-kenet.txt");
            Assert.AreEqual(18687, corpus.SentenceCount());
            Assert.AreEqual(178658, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus21()
        {
            var corpus = new SequenceCorpus("../../../postag-kenet.txt");
            Assert.AreEqual(18687, corpus.SentenceCount());
            Assert.AreEqual(178658, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus22()
        {
            var corpus = new SequenceCorpus("../../../disambiguation-framenet.txt");
            Assert.AreEqual(2704, corpus.SentenceCount());
            Assert.AreEqual(19286, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus23()
        {
            var corpus = new SequenceCorpus("../../../metamorpheme-framenet.txt");
            Assert.AreEqual(2704, corpus.SentenceCount());
            Assert.AreEqual(19286, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus24()
        {
            var corpus = new SequenceCorpus("../../../postag-framenet.txt");
            Assert.AreEqual(2704, corpus.SentenceCount());
            Assert.AreEqual(19286, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus25()
        {
            var corpus = new SequenceCorpus("../../../semanticrolelabeling-framenet.txt");
            Assert.AreEqual(2704, corpus.SentenceCount());
            Assert.AreEqual(19286, corpus.NumberOfWords());
        }

        [Test]
        public void TestCorpus26()
        {
            var corpus = new SequenceCorpus("../../../sentiment-tourism.txt");
            Assert.AreEqual(19830, corpus.SentenceCount());
            Assert.AreEqual(91152, corpus.NumberOfWords());
        }
    }
}