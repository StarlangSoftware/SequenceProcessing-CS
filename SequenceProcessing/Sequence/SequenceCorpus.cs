using System.Collections.Generic;
using System.IO;
using Corpus;
using Dictionary.Dictionary;
using Math;

namespace SequenceProcessing.Sequence
{
    public class SequenceCorpus : Corpus.Corpus
    {
        public SequenceCorpus(string fileName)
        {
            string word;
            VectorizedWord newWord;
            Sentence newSentence = null;
            this.fileName = fileName;
            var streamReader = new StreamReader(fileName);
            var line = streamReader.ReadLine();
            while (line != null)
            {
                string[] items = line.Split(" ");
                word = items[0];
                if (word == "<S>") {
                    if (items.Length == 2){
                        newSentence = new LabelledSentence(items[1]);
                    } else {
                        newSentence = new Sentence();
                    }
                } else {
                    if (word == "</S>") {
                        AddSentence(newSentence);
                    } else {
                        if (items.Length == 2) {
                            newWord = new LabelledVectorizedWord(word, items[1]);
                        } else {
                            newWord = new VectorizedWord(word, new Vector(300,0));
                        }
                        if (newSentence != null){
                            newSentence.AddWord(newWord);
                        }
                    }
                }
                line = streamReader.ReadLine();
            }
        }

        public List<string> GetClassLabels()
        {
            bool sentenceLabelled = false;
            List<string> classLabels = new List<string>();
            if (sentences[0] is LabelledSentence){
                sentenceLabelled = true;
            }
            for (int i = 0; i < SentenceCount(); i++) {
                if (sentenceLabelled){
                    LabelledSentence sentence = (LabelledSentence) sentences[i];
                    if (!classLabels.Contains(sentence.GetClassLabel())) {
                        classLabels.Add(sentence.GetClassLabel());
                    }
                } else {
                    Sentence sentence = sentences[i];
                    for (int j = 0; j < sentence.WordCount(); j++){
                        LabelledVectorizedWord word = (LabelledVectorizedWord) sentence.GetWord(j);
                        if (!classLabels.Contains(word.GetClassLabel())) {
                            classLabels.Add(word.GetClassLabel());
                        }
                    }
                }
            }
            return classLabels;
        }
    }
}