using Corpus;

namespace SequenceProcessing.Sequence
{
    public class LabelledSentence : Sentence
    {
        private string _classLabel;
        
        public LabelledSentence(string classLabel) : base()
        {
            _classLabel = classLabel;
        }

        public string GetClassLabel()
        {
            return _classLabel;
        }
    }
}