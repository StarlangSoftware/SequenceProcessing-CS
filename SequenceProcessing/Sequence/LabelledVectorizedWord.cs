using Dictionary.Dictionary;
using Math;

namespace SequenceProcessing.Sequence
{
    public class LabelledVectorizedWord : VectorizedWord
    {
        private string _classLabel;
        
        public LabelledVectorizedWord(string word, Vector embedding, string classLabel) : base(word, embedding)
        {
            _classLabel = classLabel;
        }

        public LabelledVectorizedWord(string word, string classLabel) : base(word, new Vector(300, 0))
        {
            _classLabel = classLabel;
        }

        public string GetClassLabel()
        {
            return _classLabel;
        }
    }
}