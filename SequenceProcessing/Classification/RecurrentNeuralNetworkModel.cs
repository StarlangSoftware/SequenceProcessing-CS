using System.Collections.Generic;
using Classification.Parameter;
using Math;
using SequenceProcessing.Sequence;

namespace SequenceProcessing.Classification {
    
    public class RecurrentNeuralNetworkModel : Model {
        
        public RecurrentNeuralNetworkModel(SequenceCorpus corpus, DeepNetworkParameter parameters) : base(corpus, parameters) {
            var epoch = parameters.GetEpoch(); 
            var learningRate = parameters.GetLearningRate(); 
            for (var i = 0; i < epoch; i++) { 
                corpus.ShuffleSentences(parameters.GetSeed()); 
                for (var j = 0; j < corpus.SentenceCount(); j++) { 
                    var sentence = corpus.GetSentence(j); 
                    for (var k = 0; k < sentence.WordCount(); k++) { 
                        var word = (LabelledVectorizedWord) sentence.GetWord(k); 
                        CalculateOutput(word); 
                        var rMinusY = CalculateRMinusY(word); 
                        rMinusY.MultiplyWithConstant(learningRate); 
                        var deltaWeights = new List<Matrix>(); 
                        var deltaRecurrentWeights = new List<Matrix>(); 
                        deltaWeights.Add(rMinusY.Multiply(_layers[_layers.Count - 2].Transpose())); 
                        deltaWeights.Add(rMinusY); 
                        deltaRecurrentWeights.Add(rMinusY); 
                        for (var l = parameters.LayerSize() - 1; l >= 0; l--) { 
                            var delta = deltaWeights[deltaWeights.Count - 1].Transpose().Multiply(_weights[l + 1].Partial(0, _weights[l + 1].GetRow() - 1, 0, _weights[l + 1].GetColumn() - 2)).ElementProduct(Derivative(_layers[l + 1].Partial(0, _layers[l + 1].GetRow() - 2, 0, _layers[l + 1].GetColumn() - 1), _activationFunction).Transpose()).Transpose(); 
                            deltaWeights[deltaWeights.Count - 1] = delta.Multiply(_layers[l].Transpose()); 
                            deltaRecurrentWeights[deltaRecurrentWeights.Count - 1] = delta.Multiply(_oldLayers[l].Transpose()); 
                            if (l > 0) { 
                                deltaWeights.Add(delta); 
                                deltaRecurrentWeights.Add(delta); 
                            } 
                        } 
                        _weights[_weights.Count - 1].Add(deltaWeights[0]); 
                        deltaWeights.RemoveAt(0); 
                        for (var l = 0; l < deltaWeights.Count; l++) { 
                            _weights[_weights.Count - l - 2].Add(deltaWeights[l]); 
                            _recurrentWeights[_recurrentWeights.Count - l - 1].Add(deltaRecurrentWeights[l]); 
                        } 
                        Clear(); 
                    }
                    ClearOldValues(); 
                } 
                learningRate *= parameters.GetEtaDecrease(); 
            } 
        }

        protected override void CalculateOutput(LabelledVectorizedWord word) {
            CreateInputVector(word);
            for (var l = 0; l < _layers.Count - 2; l++) {
                _layers[l + 1].Add(_recurrentWeights[l].Multiply(_oldLayers[l]));
                _layers[l + 1].Add(_weights[l].Multiply(_layers[l]));
                _layers[l + 1] = ActivationFunction(_layers[l + 1], _activationFunction);
                _layers[l + 1] = Biased(_layers[l + 1]);
            }
            _layers[_layers.Count - 1].Add(_weights[_weights.Count - 1].Multiply(_layers[_layers.Count - 2]));
            NormalizeOutput();
        }
    }
}