using System;
using System.Collections.Generic;
using Classification.Parameter;
using Corpus;
using Math;
using SequenceProcessing.Sequence;

namespace SequenceProcessing.Classification {
    
    public abstract class Model {
        
        protected SequenceCorpus _corpus;
        protected List<Matrix> _layers;
        protected List<Matrix> _oldLayers;
        protected List<Matrix> _weights;
        protected List<Matrix> _recurrentWeights;
        protected List<string> _classLabels;
        protected ActivationFunction _activationFunction;
        
        public Model(SequenceCorpus corpus, DeepNetworkParameter parameters) {
            _corpus = corpus;
            _activationFunction = parameters.GetActivationFunction();
            var layers = new List<Matrix>();
            var oldLayers = new List<Matrix>();
            var weights = new List<Matrix>();
            var recurrentWeights = new List<Matrix>();
            _classLabels = corpus.GetClassLabels();
            var inputSize = ((LabelledVectorizedWord) corpus.GetSentence(0).GetWord(0)).GetVector().Size();
            layers.Add(new Matrix(inputSize, 1));
            for (var i = 0; i < parameters.LayerSize(); i++) {
                oldLayers.Add(new Matrix(parameters.GetHiddenNodes(i), 1));
                layers.Add(new Matrix(parameters.GetHiddenNodes(i), 1));
                recurrentWeights.Add(new Matrix(parameters.GetHiddenNodes(i), parameters.GetHiddenNodes(i), -0.01, +0.01, new Random(parameters.GetSeed())));
            }
            layers.Add(new Matrix(_classLabels.Count, 1));
            for (var i = 0; i < layers.Count - 1; i++) {
                weights.Add(new Matrix(layers[i + 1].GetRow(), layers[i].GetRow() + 1, -0.01, +0.01, new Random(parameters.GetSeed())));
            }
            _layers = layers;
            _oldLayers = oldLayers;
            _weights = weights;
            _recurrentWeights = recurrentWeights;
        }
        
        protected void CreateInputVector(LabelledVectorizedWord word) {
            for (var i = 0; i < _layers[0].GetRow(); i++) {
                _layers[0].SetValue(i, 0, word.GetVector().GetValue(i));
            }
            _layers[0] = Biased(_layers[0]);
        }
        
        protected Matrix Biased(Matrix m) {
            var v = new Matrix(m.GetRow() + 1, m.GetColumn());
            for (var i = 0; i < m.GetRow(); i++) {
                v.SetValue(i, 0, m.GetValue(i, 0));
            }
            v.SetValue(m.GetRow(), 0, 1.0);
            return v;
        }
        
        protected void OldLayersUpdate() {
            for (var i = 0; i < _oldLayers.Count; i++) {
                for (var j = 0; j < _oldLayers[i].GetRow(); j++) {
                    _oldLayers[i].SetValue(j, 0, _layers[i + 1].GetValue(j, 0));
                }
            }
        }

        protected void SetLayersValuesToZero() {
            for (var j = 0; j < _layers.Count - 1; j++) {
                var size = _layers[j].GetRow();
                _layers[j] = new Matrix(size - 1, 1);
                for (var i = 0; i < _layers[j].GetRow(); i++) {
                    _layers[j].SetValue(i, 0, 0.0);
                }
            }
            for (var i = 0; i < _layers[_layers.Count - 1].GetRow(); i++) {
                _layers[_layers.Count - 1].SetValue(i, 0, 0.0);
            }
        }
        
        protected Matrix CalculateOneMinusMatrix(Matrix hidden) {
            var oneMinus = new Matrix(hidden.GetRow(), 1);
            for (var i = 0; i < oneMinus.GetRow(); i++) {
                oneMinus.SetValue(i, 0, 1 - hidden.GetValue(i, 0));
            }
            return oneMinus;
        }

        protected void NormalizeOutput() {
            var sum = 0.0;
            var values = new double[_layers[_layers.Count - 1].GetRow()];
            for (var i = 0; i < values.Length; i++) {
                sum += System.Math.Exp(_layers[_layers.Count - 1].GetValue(i, 0));
            }
            for (var i = 0; i < values.Length; i++) {
                values[i] = System.Math.Exp(_layers[_layers.Count - 1].GetValue(i, 0)) / sum;
            }
            for (var i = 0; i < values.Length; i++) {
                _layers[_layers.Count - 1].SetValue(i, 0, values[i]);
            }
        }
        
        protected Matrix CalculateRMinusY(LabelledVectorizedWord word) {
            var r = new Matrix(_classLabels.Count, 1);
            var index = _classLabels.IndexOf(word.GetClassLabel());
            r.SetValue(index, 0, 1.0);
            for (var i = 0; i < _classLabels.Count; i++) {
                r.SetValue(i, 0, r.GetValue(i, 0) - _layers[_layers.Count - 1].GetValue(i, 0));
            }
            return r;
        }

        protected Matrix Derivative(Matrix matrix, ActivationFunction function) {
            if (function.Equals(global::Classification.Parameter.ActivationFunction.SIGMOID)) {
                var oneMinusHidden = CalculateOneMinusMatrix(matrix);
                return matrix.ElementProduct(oneMinusHidden);
            } else if (function.Equals(global::Classification.Parameter.ActivationFunction.TANH)) {
                var oneMinusA2 = new Matrix(matrix.GetRow(), 1);
                var a2 = matrix.ElementProduct(matrix);
                for (var i = 0; i < oneMinusA2.GetRow(); i++) {
                    oneMinusA2.SetValue(i, 0, 1.0 - a2.GetValue(i, 0));
                }
                return oneMinusA2;
            } else {
                var der = new Matrix(matrix.GetRow(), 1);
                for (var i = 0; i < matrix.GetRow(); i++) {
                    if (matrix.GetValue(i, 0) > 0) {
                        der.SetValue(i, 0, 1.0);
                    }
                }
                return der;
            }
        }
        
        protected Matrix ActivationFunction(Matrix matrix, ActivationFunction function) {
            var r = new Matrix(matrix.GetRow(), matrix.GetColumn());
            if (function.Equals(global::Classification.Parameter.ActivationFunction.SIGMOID)) {
                for (var i = 0; i < matrix.GetRow(); i++) {
                    r.SetValue(i, 0, 1 / (1 + System.Math.Exp(-matrix.GetValue(i, 0))));
                }
            } else if (function.Equals(global::Classification.Parameter.ActivationFunction.TANH)) {
                for (var i = 0; i < matrix.GetRow(); i++) {
                    r.SetValue(i, 0, System.Math.Tanh(matrix.GetValue(i, 0)));
                }
            } else {
                for (var i = 0; i < matrix.GetRow(); i++) {
                    if (matrix.GetValue(i, 0) < 0) {
                        r.SetValue(i, 0, 0.0);
                    } else {
                        r.SetValue(i, 0, matrix.GetValue(i, 0));
                    }
                }
            }
            return r;
        }

        protected abstract void Clear();

        protected void ClearOldValues() {
            for (var i = 0; i < _oldLayers.Count; i++) {
                for (var k = 0; k < _oldLayers[i].GetRow(); k++) {
                    _oldLayers[i].SetValue(k, 0, 0.0);
                }
            }
        }
        
        protected abstract void CalculateOutput(LabelledVectorizedWord word);

        public List<string> Predict(Sentence sentence) {
            var classLabels = new List<string>();
            for (var i = 0; i < sentence.WordCount(); i++) {
                var word = (LabelledVectorizedWord) sentence.GetWord(i);
                CalculateOutput(word);
                var bestValue = Double.MinValue;
                var best = _classLabels[0];
                for (var j = 0; j < _layers[_layers.Count - 1].GetRow(); j++) {
                    if (_layers[_layers.Count - 1].GetValue(j, 0) > bestValue) {
                        bestValue = _layers[_layers.Count - 1].GetValue(j, 0);
                        best = _classLabels[j];
                    }
                }
                classLabels.Add(best);
                Clear();
            }
            ClearOldValues();
            return classLabels;
        }
    }
}