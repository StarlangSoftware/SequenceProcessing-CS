using System;
using System.Collections.Generic;
using Classification.Parameter;
using Math;
using SequenceProcessing.Sequence;

namespace SequenceProcessing.Classification {
    
    public class LongShortTermMemoryModel : Model {
        
        private List<Matrix> _fVectors;
        private List<Matrix> _fWeights;
        private List<Matrix> _fRecurrentWeights;
        private List<Matrix> _gVectors;
        private List<Matrix> _gWeights;
        private List<Matrix> _gRecurrentWeights;
        private List<Matrix> _iVectors;
        private List<Matrix> _iWeights;
        private List<Matrix> _iRecurrentWeights;
        private List<Matrix> _oVectors;
        private List<Matrix> _oWeights;
        private List<Matrix> _oRecurrentWeights;
        private List<Matrix> _cVectors;
        private List<Matrix> _cOldVectors;
        
        public LongShortTermMemoryModel(SequenceCorpus corpus, DeepNetworkParameter parameters, Initializer.Initializer initializer) : base(corpus, parameters, initializer) {
             var epoch = parameters.GetEpoch(); 
             var learningRate = parameters.GetLearningRate(); 
             _fVectors = new List<Matrix>(); 
             _fWeights = new List<Matrix>(); 
             _fRecurrentWeights = new List<Matrix>(); 
             _gVectors = new List<Matrix>(); 
             _gWeights = new List<Matrix>(); 
             _gRecurrentWeights = new List<Matrix>(); 
             _iVectors = new List<Matrix>(); 
             _iWeights = new List<Matrix>(); 
             _iRecurrentWeights = new List<Matrix>(); 
             _oVectors = new List<Matrix>(); 
             _oWeights = new List<Matrix>(); 
             _oRecurrentWeights = new List<Matrix>(); 
             _cVectors = new List<Matrix>(); 
             _cOldVectors = new List<Matrix>(); 
             for (var i = 0; i < parameters.LayerSize(); i++) { 
                 _fVectors.Add(new Matrix(parameters.GetHiddenNodes(i), 1)); 
                 _gVectors.Add(new Matrix(parameters.GetHiddenNodes(i), 1)); 
                 _iVectors.Add(new Matrix(parameters.GetHiddenNodes(i), 1)); 
                 _oVectors.Add(new Matrix(parameters.GetHiddenNodes(i), 1)); 
                 _cVectors.Add(new Matrix(parameters.GetHiddenNodes(i), 1)); 
                 _cOldVectors.Add(new Matrix(parameters.GetHiddenNodes(i), 1)); 
                 _fWeights.Add(initializer.Initialize(_layers[i + 1].GetRow(), _layers[i].GetRow() + 1, new Random(parameters.GetSeed()))); 
                 _gWeights.Add(initializer.Initialize(_layers[i + 1].GetRow(), _layers[i].GetRow() + 1, new Random(parameters.GetSeed()))); 
                 _iWeights.Add(initializer.Initialize(_layers[i + 1].GetRow(), _layers[i].GetRow() + 1, new Random(parameters.GetSeed()))); 
                 _oWeights.Add(initializer.Initialize(_layers[i + 1].GetRow(), _layers[i].GetRow() + 1, new Random(parameters.GetSeed()))); 
                 _fRecurrentWeights.Add(initializer.Initialize(parameters.GetHiddenNodes(i), parameters.GetHiddenNodes(i), new Random(parameters.GetSeed()))); 
                 _gRecurrentWeights.Add(initializer.Initialize(parameters.GetHiddenNodes(i), parameters.GetHiddenNodes(i), new Random(parameters.GetSeed()))); 
                 _iRecurrentWeights.Add(initializer.Initialize(parameters.GetHiddenNodes(i), parameters.GetHiddenNodes(i), new Random(parameters.GetSeed()))); 
                 _oRecurrentWeights.Add(initializer.Initialize(parameters.GetHiddenNodes(i), parameters.GetHiddenNodes(i), new Random(parameters.GetSeed()))); 
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
                         var deltaWeight = rMinusY.Multiply(_layers[_layers.Count - 2].Transpose()); 
                         var fDeltaWeights = new List<Matrix>(); 
                         var fDeltaRecurrentWeights = new List<Matrix>(); 
                         var gDeltaWeights = new List<Matrix>(); 
                         var gDeltaRecurrentWeights = new List<Matrix>(); 
                         var iDeltaWeights = new List<Matrix>(); 
                         var iDeltaRecurrentWeights = new List<Matrix>(); 
                         var oDeltaWeights = new List<Matrix>(); 
                         var oDeltaRecurrentWeights = new List<Matrix>(); 
                         fDeltaWeights.Add(rMinusY.Transpose().Multiply(_weights[_weights.Count - 1].Partial(0, _weights[_weights.Count - 1].GetRow() - 1, 0, _weights[_weights.Count - 1].GetColumn() - 2)).Transpose()); 
                         fDeltaRecurrentWeights.Add((Matrix) fDeltaWeights[0].Clone()); 
                         gDeltaWeights.Add((Matrix) fDeltaWeights[0].Clone()); 
                         gDeltaRecurrentWeights.Add((Matrix) fDeltaWeights[0].Clone()); 
                         iDeltaWeights.Add((Matrix) fDeltaWeights[0].Clone()); 
                         iDeltaRecurrentWeights.Add((Matrix) fDeltaWeights[0].Clone()); 
                         oDeltaWeights.Add((Matrix) fDeltaWeights[0].Clone()); 
                         oDeltaRecurrentWeights.Add((Matrix) fDeltaWeights[0].Clone()); 
                         for (var l = parameters.LayerSize() - 1; l >= 0; l--) { 
                             var cTanH = ActivationFunction(_cVectors[l], global::Classification.Parameter.ActivationFunction.TANH); 
                             var cDerivative = Derivative(cTanH, global::Classification.Parameter.ActivationFunction.TANH); 
                             var fDelta = fDeltaWeights[fDeltaWeights.Count - 1].ElementProduct(_oVectors[l].ElementProduct(cDerivative)).ElementProduct(_cOldVectors[l]).ElementProduct(Derivative(_fVectors[l], _activationFunction)); 
                             var gDelta = gDeltaWeights[gDeltaWeights.Count - 1].ElementProduct(_oVectors[l].ElementProduct(cDerivative)).ElementProduct(_iVectors[l]).ElementProduct(Derivative(_gVectors[l], global::Classification.Parameter.ActivationFunction.TANH)); 
                             var iDelta = iDeltaWeights[iDeltaWeights.Count - 1].ElementProduct(_oVectors[l].ElementProduct(cDerivative)).ElementProduct(_gVectors[l]).ElementProduct(Derivative(_iVectors[l], _activationFunction)); 
                             var oDelta = oDeltaWeights[oDeltaWeights.Count - 1].ElementProduct(cTanH).ElementProduct(Derivative(_oVectors[l], _activationFunction)); 
                             fDeltaWeights[fDeltaWeights.Count - 1] = fDelta.Multiply(_layers[l].Transpose()); 
                             fDeltaRecurrentWeights[fDeltaRecurrentWeights.Count - 1] = fDelta.Multiply(_oldLayers[l].Transpose()); 
                             gDeltaWeights[gDeltaWeights.Count - 1] = gDelta.Multiply(_layers[l].Transpose()); 
                             gDeltaRecurrentWeights[gDeltaRecurrentWeights.Count - 1] = gDelta.Multiply(_oldLayers[l].Transpose()); 
                             iDeltaWeights[iDeltaWeights.Count - 1] = iDelta.Multiply(_layers[l].Transpose()); 
                             iDeltaRecurrentWeights[iDeltaRecurrentWeights.Count - 1] = iDelta.Multiply(_oldLayers[l].Transpose()); 
                             oDeltaWeights[oDeltaWeights.Count - 1] = oDelta.Multiply(_layers[l].Transpose()); 
                             oDeltaRecurrentWeights[oDeltaRecurrentWeights.Count - 1] = oDelta.Multiply(_oldLayers[l].Transpose()); 
                             if (l > 0) { 
                                 fDeltaWeights.Add(fDelta.Transpose().Multiply(_fWeights[l].Partial(0, _fWeights[l].GetRow() - 1, 0, _fWeights[l].GetColumn() - 2)).Transpose()); 
                                 fDeltaRecurrentWeights.Add(fDelta.Transpose().Multiply(_fWeights[l].Partial(0, _fWeights[l].GetRow() - 1, 0, _fWeights[l].GetColumn() - 2)).Transpose()); 
                                 gDeltaWeights.Add(gDelta.Transpose().Multiply(_gWeights[l].Partial(0, _gWeights[l].GetRow() - 1, 0, _gWeights[l].GetColumn() - 2)).Transpose()); 
                                 gDeltaRecurrentWeights.Add(gDelta.Transpose().Multiply(_gWeights[l].Partial(0, _gWeights[l].GetRow() - 1, 0, _gWeights[l].GetColumn() - 2)).Transpose()); 
                                 iDeltaWeights.Add(iDelta.Transpose().Multiply(_iWeights[l].Partial(0, _iWeights[l].GetRow() - 1, 0, _iWeights[l].GetColumn() - 2)).Transpose()); 
                                 iDeltaRecurrentWeights.Add(iDelta.Transpose().Multiply(_iWeights[l].Partial(0, _iWeights[l].GetRow() - 1, 0, _iWeights[l].GetColumn() - 2)).Transpose()); 
                                 oDeltaWeights.Add(oDelta.Transpose().Multiply(_oWeights[l].Partial(0, _oWeights[l].GetRow() - 1, 0, _oWeights[l].GetColumn() - 2)).Transpose()); 
                                 oDeltaRecurrentWeights.Add(oDelta.Transpose().Multiply(_oWeights[l].Partial(0, _oWeights[l].GetRow() - 1, 0, _oWeights[l].GetColumn() - 2)).Transpose()); 
                             } 
                         } 
                         _weights[_weights.Count - 1].Add(deltaWeight); 
                         for (var l = 0; l < fDeltaWeights.Count; l++) { 
                             _fWeights[_fWeights.Count - l - 1].Add(fDeltaWeights[l]); 
                             _gWeights[_gWeights.Count - l - 1].Add(gDeltaWeights[l]); 
                             _iWeights[_iWeights.Count - l - 1].Add(iDeltaWeights[l]); 
                             _oWeights[_oWeights.Count - l - 1].Add(oDeltaWeights[l]); 
                             _fRecurrentWeights[_fRecurrentWeights.Count - l - 1].Add(fDeltaRecurrentWeights[l]); 
                             _gRecurrentWeights[_gRecurrentWeights.Count - l - 1].Add(gDeltaRecurrentWeights[l]); 
                             _iRecurrentWeights[_iRecurrentWeights.Count - l - 1].Add(iDeltaRecurrentWeights[l]); 
                             _oRecurrentWeights[_oRecurrentWeights.Count - l - 1].Add(oDeltaRecurrentWeights[l]); 
                         } 
                         Clear(); 
                     } 
                     ClearOldValues(); 
                 } 
                 learningRate *= parameters.GetEtaDecrease(); 
             } 
        }

        protected override void Clear() {
            OldLayersUpdate();
            SetLayersValuesToZero();
            for (var l = 0; l < _layers.Count - 2; l++) {
                for (var m = 0; m < _fVectors[l].GetRow(); m++) {
                    _fVectors[l].SetValue(m, 0, 0.0);
                    _gVectors[l].SetValue(m, 0, 0.0);
                    _iVectors[l].SetValue(m, 0, 0.0);
                    _oVectors[l].SetValue(m, 0, 0.0);
                    _cVectors[l].SetValue(m, 0, 0.0);
                }
            }
        }

        protected override void CalculateOutput(LabelledVectorizedWord word) {
            CreateInputVector(word);
            var kVectors = new List<Matrix>();
            var jVectors = new List<Matrix>();
            for (var i = 0; i < _layers.Count - 2; i++) {
                _fVectors[i].Add(_fRecurrentWeights[i].Multiply(_oldLayers[i]).Sum(_fWeights[i].Multiply(_layers[i])));
                _fVectors[i] = ActivationFunction(_fVectors[i], _activationFunction);
                kVectors.Add(_cOldVectors[i].ElementProduct(_fVectors[i]));
                _gVectors[i].Add(_gRecurrentWeights[i].Multiply(_oldLayers[i]).Sum(_gWeights[i].Multiply(_layers[i])));
                _gVectors[i] = ActivationFunction(_gVectors[i], global::Classification.Parameter.ActivationFunction.TANH);
                _iVectors[i].Add(_iRecurrentWeights[i].Multiply(_oldLayers[i]).Sum(_iWeights[i].Multiply(_layers[i])));
                _iVectors[i] = ActivationFunction(_iVectors[i], _activationFunction);
                jVectors.Add(_gVectors[i].ElementProduct(_iVectors[i]));
                _cVectors[i].Add(jVectors[i].Sum(kVectors[i]));
                _oVectors[i].Add(_oRecurrentWeights[i].Multiply(_oldLayers[i]).Sum(_oWeights[i].Multiply(_layers[i])));
                _oVectors[i] = ActivationFunction(_oVectors[i], _activationFunction);
                _layers[i + 1].Add(_oVectors[i].ElementProduct(ActivationFunction(_cVectors[i], global::Classification.Parameter.ActivationFunction.TANH)));
                _layers[i + 1] = Biased(_layers[i + 1]);
            }
            _layers[_layers.Count - 1].Add(_weights[_weights.Count - 1].Multiply(_layers[_layers.Count - 2]));
            NormalizeOutput();
        }
        
        protected new void OldLayersUpdate() {
            for (var i = 0; i < _oldLayers.Count; i++) {
                for (var j = 0; j < _oldLayers[i].GetRow(); j++) {
                    _oldLayers[i].SetValue(j, 0, _layers[i + 1].GetValue(j, 0));
                    _cOldVectors[i].SetValue(j, 0, _cVectors[i].GetValue(j, 0));
                }
            }
        }
        
        protected new void ClearOldValues() {
            for (var i = 0; i < _oldLayers.Count; i++) {
                for (var k = 0; k < _oldLayers[i].GetRow(); k++) {
                    _cOldVectors[i].SetValue(k, 0, 0.0);
                    _oldLayers[i].SetValue(k, 0, 0.0);
                }
            }
        }
    }
}