using Math;

namespace SequenceProcessing.Initializer {
    
    public class UniformXavier : Initializer {
        
        public Matrix Initialize(int row, int col, System.Random random) {
            var m = new Matrix(row, col);
            for (var i = 0; i < row; i++) {
                for (var j = 0; j < col; j++) {
                    m.SetValue(i, j, (2 * random.NextDouble() - 1) * System.Math.Sqrt(6.0 / (row + col)));
                }
            }
            return m;
        }
    }
}