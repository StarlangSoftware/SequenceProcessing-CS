using NUnit.Framework;
using SequenceProcessing.Sequence;

namespace Test
{
    public class SequenceCorpusTest
    {
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