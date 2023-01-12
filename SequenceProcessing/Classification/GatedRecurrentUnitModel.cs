using System;
using System.Collections.Generic;
using Classification.Parameter;
using Math;
using SequenceProcessing.Sequence;

namespace SequenceProcessing.Classification {
    
    public class GatedRecurrentUnitModel : Model {
        
        private List<Matrix> _aVectors;
        private List<Matrix> _zVectors;
        private List<Matrix> _rVectors;
        private List<Matrix> _zWeights;
        private List<Matrix> _zRecurrentWeights;
        private List<Matrix> _rWeights;
        private List<Matrix> _rRecurrentWeights;


        public GatedRecurrentUnitModel(SequenceCorpus corpus, DeepNetworkParameter parameters) : base(corpus, parameters) {
            var epoch = parameters.GetEpoch(); 
            var learningRate = parameters.GetLearningRate(); 
            _aVectors = new List<Matrix>(); 
            _zVectors = new List<Matrix>(); 
            _rVectors = new List<Matrix>(); 
            _zWeights = new List<Matrix>(); 
            _zRecurrentWeights = new List<Matrix>(); 
            _rWeights = new List<Matrix>(); 
            _rRecurrentWeights = new List<Matrix>(); 
            for (var i = 0; i < parameters.LayerSize(); i++) { 
                _aVectors.Add(new Matrix(parameters.GetHiddenNodes(i), 1)); 
                _zVectors.Add(new Matrix(parameters.GetHiddenNodes(i), 1)); 
                _rVectors.Add(new Matrix(parameters.GetHiddenNodes(i), 1)); 
                _zWeights.Add(new Matrix(_layers[i + 1].GetRow(), _layers[i].GetRow() + 1, -0.01, +0.01, new Random(parameters.GetSeed()))); 
                _rWeights.Add(new Matrix(_layers[i + 1].GetRow(), _layers[i].GetRow() + 1, -0.01, +0.01, new Random(parameters.GetSeed()))); 
                _zRecurrentWeights.Add(new Matrix(parameters.GetHiddenNodes(i), parameters.GetHiddenNodes(i), -0.01, +0.01, new Random(parameters.GetSeed()))); 
                _rRecurrentWeights.Add(new Matrix(parameters.GetHiddenNodes(i), parameters.GetHiddenNodes(i), -0.01, +0.01, new Random(parameters.GetSeed()))); 
            } 
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
                        var rDeltaWeights = new List<Matrix>(); 
                        var rDeltaRecurrentWeights = new List<Matrix>(); 
                        var zDeltaWeights = new List<Matrix>(); 
                        var zDeltaRecurrentWeights = new List<Matrix>(); 
                        deltaWeights.Add(rMinusY.Multiply(_layers[_layers.Count - 2].Transpose())); 
                        deltaWeights.Add(rMinusY.Transpose().Multiply(_weights[_weights.Count - 1].Partial(0, _weights[_weights.Count - 1].GetRow() - 1, 0, _weights[_weights.Count - 1].GetColumn() - 2)).Transpose()); 
                        deltaRecurrentWeights.Add((Matrix) deltaWeights[deltaWeights.Count - 1].Clone()); 
                        rDeltaWeights.Add((Matrix) deltaWeights[deltaWeights.Count - 1].Clone()); 
                        rDeltaRecurrentWeights.Add((Matrix) deltaWeights[deltaWeights.Count - 1].Clone()); 
                        zDeltaWeights.Add((Matrix) deltaWeights[deltaWeights.Count - 1].Clone()); 
                        zDeltaRecurrentWeights.Add((Matrix) deltaWeights[deltaWeights.Count - 1].Clone()); 
                        for (var l = parameters.LayerSize() - 1; l >= 0; l--) { 
                            var delta = deltaWeights[deltaWeights.Count - 1].ElementProduct(_zVectors[l]).ElementProduct(Derivative(_aVectors[l], global::Classification.Parameter.ActivationFunction.TANH)); 
                            var diff = (Matrix) _aVectors[l].Clone(); 
                            diff.Subtract(_oldLayers[l]); 
                            var zDelta = zDeltaWeights[zDeltaWeights.Count - 1].ElementProduct(diff).ElementProduct(Derivative(_zVectors[l], _activationFunction)); 
                            var rDelta = rDeltaWeights[rDeltaWeights.Count - 1].ElementProduct(diff).ElementProduct(Derivative(_zVectors[l], _activationFunction)).Transpose().Multiply(_recurrentWeights[l]).Transpose().ElementProduct(_oldLayers[l]).ElementProduct(Derivative(_rVectors[l], _activationFunction)); 
                            deltaWeights[deltaWeights.Count - 1] = delta.Multiply(_layers[l].Transpose()); 
                            deltaRecurrentWeights[deltaRecurrentWeights.Count - 1] = delta.Multiply((_rVectors[l].ElementProduct(_oldLayers[l])).Transpose()); 
                            zDeltaWeights[zDeltaWeights.Count - 1] = zDelta.Multiply(_layers[l].Transpose()); 
                            zDeltaRecurrentWeights[zDeltaRecurrentWeights.Count - 1] = zDelta.Multiply(_oldLayers[l].Transpose()); 
                            rDeltaWeights[rDeltaWeights.Count - 1] = rDelta.Multiply(_layers[l].Transpose()); 
                            rDeltaRecurrentWeights[rDeltaRecurrentWeights.Count - 1] = rDelta.Multiply(_oldLayers[l].Transpose()); 
                            if (l > 0) { 
                                deltaWeights.Add(delta.Transpose().Multiply(_weights[l].Partial(0, _weights[l].GetRow() - 1, 0, _weights[l].GetColumn() - 2)).Transpose()); 
                                deltaRecurrentWeights.Add(delta.Transpose().Multiply(_weights[l].Partial(0, _weights[l].GetRow() - 1, 0, _weights[l].GetColumn() - 2)).Transpose()); 
                                zDeltaWeights.Add(zDelta.Transpose().Multiply(_zWeights[l].Partial(0, _zWeights[l].GetRow() - 1, 0, _zWeights[l].GetColumn() - 2)).Transpose()); 
                                zDeltaRecurrentWeights.Add(zDelta.Transpose().Multiply(_zWeights[l].Partial(0, _zWeights[l].GetRow() - 1, 0, _zWeights[l].GetColumn() - 2)).Transpose()); 
                                rDeltaWeights.Add(rDelta.Transpose().Multiply(_rWeights[l].Partial(0, _rWeights[l].GetRow() - 1, 0, _rWeights[l].GetColumn() - 2)).Transpose()); 
                                rDeltaRecurrentWeights.Add(rDelta.Transpose().Multiply(_rWeights[l].Partial(0, _rWeights[l].GetRow() - 1, 0, _rWeights[l].GetColumn() - 2)).Transpose()); 
                            } 
                        } 
                        _weights[_weights.Count - 1].Add(deltaWeights[0]); 
                        deltaWeights.RemoveAt(0); 
                        for (var l = 0; l < deltaWeights.Count; l++) { 
                            _weights[_weights.Count - l - 2].Add(deltaWeights[l]); 
                            _rWeights[_rWeights.Count - l - 1].Add(rDeltaWeights[l]); 
                            _zWeights[_zWeights.Count - l - 1].Add(zDeltaWeights[l]); 
                            _recurrentWeights[_recurrentWeights.Count - l - 1].Add(deltaRecurrentWeights[l]); 
                            _zRecurrentWeights[_zRecurrentWeights.Count - l - 1].Add(zDeltaRecurrentWeights[l]); 
                            _rRecurrentWeights[_rRecurrentWeights.Count - l - 1].Add(rDeltaRecurrentWeights[l]); 
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
                _rVectors[l].Add(_rWeights[l].Multiply(_layers[l]));
                _zVectors[l].Add(_zWeights[l].Multiply(_layers[l]));
                _rVectors[l].Add(_rRecurrentWeights[l].Multiply(_oldLayers[l]));
                _zVectors[l].Add(_zRecurrentWeights[l].Multiply(_oldLayers[l]));
                _rVectors[l] = ActivationFunction(_rVectors[l], _activationFunction);
                _zVectors[l] = ActivationFunction(_zVectors[l], _activationFunction);
                _aVectors[l].Add(_recurrentWeights[l].Multiply(_rVectors[l].ElementProduct(_oldLayers[l])));
                _aVectors[l].Add(_weights[l].Multiply(_layers[l]));
                _aVectors[l] = ActivationFunction(_aVectors[l], global::Classification.Parameter.ActivationFunction.TANH);
                _layers[l + 1].Add(CalculateOneMinusMatrix(_zVectors[l]).ElementProduct(_oldLayers[l]));
                _layers[l + 1].Add(_zVectors[l].ElementProduct(_aVectors[l]));
                _layers[l + 1] = Biased(_layers[l + 1]);
            }
            _layers[_layers.Count - 1].Add(_weights[_weights.Count - 1].Multiply(_layers[_layers.Count - 2])); 
            NormalizeOutput();
        }
        
        protected override void Clear() {
            OldLayersUpdate();
            SetLayersValuesToZero();
            for (var l = 0; l < _layers.Count - 2; l++) {
                for (var m = 0; m < _aVectors[l].GetRow(); m++) {
                    _aVectors[l].SetValue(m, 0, 0.0);
                    _zVectors[l].SetValue(m, 0, 0.0);
                    _rVectors[l].SetValue(m, 0, 0.0);
                }
            }
        }
    }
}