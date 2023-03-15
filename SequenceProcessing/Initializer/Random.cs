using Math;

namespace SequenceProcessing.Initializer {
    
    public class Random : Initializer {
        
        public Matrix Initialize(int row, int col, System.Random random) {
            return new Matrix(row, col, -0.01, +0.01, random);
        }
    }
}