using Math;

namespace SequenceProcessing.Initializer {
    
    public interface Initializer {
        Matrix Initialize(int row, int col, System.Random random);
    }
}