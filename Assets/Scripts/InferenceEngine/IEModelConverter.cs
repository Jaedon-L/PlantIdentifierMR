using UnityEngine;
using Unity.InferenceEngine;

public class IEModelConverter : MonoBehaviour
{
    public ModelAsset _onnxModel;
    [SerializeField, Range(0, 1)] private float _iouThreshold = 0.6f;
    [SerializeField, Range(0, 1)] private float _scoreThreshold = 0.23f;
}