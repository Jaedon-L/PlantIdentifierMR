using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.InferenceEngine;
using UnityEngine;

public class IEExecutor : MonoBehaviour
{
    enum InferenceDownloadState
    {
        Running = 0,
        RequestingOutput0 = 1,
        Success = 2,
        Error = 3,
        Cleanup = 4,
        Completed = 5
    }

    [SerializeField] private Vector2Int _inputSize = new(640, 640);
    [SerializeField] private BackendType _backend = BackendType.CPU;
    [SerializeField] private ModelAsset _sentisModel;
    [SerializeField] private int _layersPerFrame = 25;
    [SerializeField] private float _confidenceThreshold = 0.5f;
    [SerializeField] private TextAsset _labelsAsset;
    [SerializeField] private Transform _displayLocation;
    public bool IsModelLoaded { get; private set; } = false;
    private List<BoundingBox> _lastDetections = new List<BoundingBox>();

    [SerializeField] private IEBoxer _ieBoxer;

    // private IEMasker _ieMasker;
    private Worker _inferenceEngineWorker;
    private IEnumerator _schedule;
    private InferenceDownloadState _downloadState = InferenceDownloadState.Running;

    private Tensor<float> _input;

    private Tensor _buffer;
    private Tensor<float> _output0BoxCoords;
    // private Tensor<int> _output1LabelIds;
    // private Tensor<float> _output2Masks;
    // private Tensor<float> _output3MaskWeights;

    private bool _started = false;
    private bool _isWaitingForReadbackRequest = false;

    private IEnumerator Start()
    {
        // Wait for the UI to be ready because when Sentis load the model it will block the main thread.
        yield return new WaitForSeconds(0.05f);

        // _ieMasker = new IEMasker(_displayLocation, _confidenceThreshold);
        LoadModel();
    }

    private void Update()
    {
        UpdateInference();
    }

    private void OnDestroy()
    {
        if (_schedule != null)
        {
            StopCoroutine(_schedule);
        }
        _input?.Dispose();
        _inferenceEngineWorker?.Dispose();
    }

    public void RunInference(Texture inputTexture)
    {
        // If the inference is not running prepare the input
        if (!_started)
        {
            // Clean last input
            _input?.Dispose();

            // Check if we have a texture from the camera
            if (!inputTexture) return;

            // Convert the texture to a Tensor and schedule the inference
            _inputSize = new Vector2Int(inputTexture.width, inputTexture.height);
            _input = TextureConverter.ToTensor(inputTexture, 640, 640, 3);
            _schedule = _inferenceEngineWorker.ScheduleIterable(_input);

            _downloadState = InferenceDownloadState.Running;
            _started = true;
        }
    }

    public bool IsRunning() => _started;

    private void LoadModel()
    {
        // Load model
        Model model = ModelLoader.Load(_sentisModel);

        // Create engine to run model
        _inferenceEngineWorker = new Worker(model, _backend);

        // Run a inference with an empty input to load the model in the memory and not pause the main thread.
        Tensor input = TextureConverter.ToTensor(new Texture2D(_inputSize.x, _inputSize.y), _inputSize.x, _inputSize.y, 3);
        _inferenceEngineWorker.Schedule(input);

        IsModelLoaded = true;
    }

    private void UpdateInference()
    {
        // Run the inference layer by layer to not block the main thread.
        if (!_started) return;
        if (_downloadState == InferenceDownloadState.Running)
        {
            int it = 0;
            while (_schedule.MoveNext()) if (++it % _layersPerFrame == 0) return;

            // If we reach here, all layers have been processed
            _downloadState = InferenceDownloadState.RequestingOutput0;
        }
        else
        {
            // Get the result once all layers are processed
            UpdateProcessInferenceResults();
        }
    }

    private void UpdateProcessInferenceResults()
    {
        switch (_downloadState)
        {
            case InferenceDownloadState.RequestingOutput0:
                if (!_isWaitingForReadbackRequest)
                {
                    _buffer = GetOutputBuffer(0);
                    InitiateReadbackRequest(_buffer);
                }
                else
                {
                    if (_buffer.IsReadbackRequestDone())
                    {
                        _output0BoxCoords = _buffer.ReadbackAndClone() as Tensor<float>;
                        _isWaitingForReadbackRequest = false;

                        if (_output0BoxCoords.shape[0] > 0)
                        {
                            Debug.Log("Sentis: _output0BoxCoords ready");
                            _downloadState = InferenceDownloadState.Success;
                        }
                        else
                        {
                            Debug.LogError("Sentis: _output0BoxCoords empty");
                            _downloadState = InferenceDownloadState.Error;
                        }
                        _buffer?.Dispose();
                    }
                }
                break;
            // case InferenceDownloadState.RequestingOutput1:
            //     if (!_isWaitingForReadbackRequest)
            //     {
            //         _buffer = GetOutputBuffer(1) as Tensor<int>;
            //         InitiateReadbackRequest(_buffer);
            //     }
            //     else
            //     {
            //         if (_buffer.IsReadbackRequestDone())
            //         {
            //             _output1LabelIds = _buffer.ReadbackAndClone() as Tensor<int>;
            //             _isWaitingForReadbackRequest = false;

            //             if (_output1LabelIds.shape[0] > 0)
            //             {
            //                 Debug.Log("Sentis: _output1LabelIds ready");
            //                 _downloadState = InferenceDownloadState.RequestingOutput2;
            //             }
            //             else
            //             {
            //                 Debug.LogError("Sentis: _output1LabelIds empty");
            //                 _downloadState = InferenceDownloadState.Error;
            //             }
            //             _buffer?.Dispose();
            //         }
            //     }
            //     break;
            // case InferenceDownloadState.RequestingOutput2:
            //     if (!_isWaitingForReadbackRequest)
            //     {
            //         _buffer = GetOutputBuffer(2) as Tensor<float>;
            //         InitiateReadbackRequest(_buffer);
            //     }
            //     else
            //     {
            //         if (_buffer.IsReadbackRequestDone())
            //         {
            //             _output2Masks = _buffer.ReadbackAndClone() as Tensor<float>;
            //             _isWaitingForReadbackRequest = false;

            //             if (_output2Masks.shape[0] > 0)
            //             {
            //                 Debug.Log("Sentis: _output2Masks ready");
            //                 _downloadState = InferenceDownloadState.RequestingOutput3;
            //             }
            //             else
            //             {
            //                 Debug.LogError("Sentis: _output2Masks empty");
            //                 _downloadState = InferenceDownloadState.Error;
            //             }
            //             _buffer?.Dispose();
            //         }
            //     }
            //     break;
            // case InferenceDownloadState.RequestingOutput3:
            //     if (!_isWaitingForReadbackRequest)
            //     {
            //         _buffer = GetOutputBuffer(3) as Tensor<float>;
            //         InitiateReadbackRequest(_buffer);
            //     }
            //     else
            //     {
            //         if (_buffer.IsReadbackRequestDone())
            //         {
            //             _output3MaskWeights = _buffer.ReadbackAndClone() as Tensor<float>;
            //             _isWaitingForReadbackRequest = false;

            //             if (_output3MaskWeights.shape[0] > 0)
            //             {
            //                 Debug.Log("Sentis: _output3MaskWeights ready");
            //                 _downloadState = InferenceDownloadState.Success;
            //             }
            //             else
            //             {
            //                 Debug.LogError("Sentis: _output3MaskWeights empty");
            //                 _downloadState = InferenceDownloadState.Error;
            //             }
            //             _buffer?.Dispose();
            //         }
            //     }
            //     break;
            case InferenceDownloadState.Success:
                List<BoundingBox> boundingBoxes = _ieBoxer.DrawBoxes(_output0BoxCoords, _inputSize.x, _inputSize.y);
                _lastDetections = boundingBoxes;
                Debug.Log($"Inference completed: {_lastDetections.Count} boxes detected.");
                // Optionally: log top few
                if (_lastDetections.Count > 0)
                {
                    var best = _lastDetections.OrderByDescending(b => b.Confidence).First();
                    Debug.Log($"Top detection: {best.ClassName} ({best.Confidence:F2})");
                }
                // _ieMasker.DrawMask(boundingBoxes, _output3MaskWeights, _inputSize.x, _inputSize.y);
                _downloadState = InferenceDownloadState.Cleanup;
                break;
            case InferenceDownloadState.Error:
                _downloadState = InferenceDownloadState.Cleanup;
                break;
            case InferenceDownloadState.Cleanup:
                _downloadState = InferenceDownloadState.Completed;
                _started = false;
                _output0BoxCoords?.Dispose();
                // _output1LabelIds?.Dispose();
                // _output2Masks?.Dispose();
                // _output3MaskWeights?.Dispose();
                break;
        }
    }

    private Tensor GetOutputBuffer(int outputIndex)
    {
        return _inferenceEngineWorker.PeekOutput(outputIndex);
    }

    private void InitiateReadbackRequest(Tensor pullTensor)
    {
        if (pullTensor.dataOnBackend != null)
        {
            pullTensor.ReadbackRequest();
            _isWaitingForReadbackRequest = true;
        }
        else
        {
            Debug.LogError($"InitiateReadbackRequest: No data output");
            _downloadState = InferenceDownloadState.Error;
        }
    }
    /// <summary>
    /// Returns the top prediction (class name + confidence) from the last inference.
    /// </summary>
    public string GetTopPrediction()
    {
        if (_lastDetections == null || _lastDetections.Count == 0)
            return "No detections";
        var best = _lastDetections.OrderByDescending(b => b.Confidence).First();
        return $"{best.ClassName}";
    }

    /// <summary>
    /// Returns up to k top predictions as a list of (className, confidence).
    /// </summary>
    public List<(string className, float confidence)> GetTopKPredictions(int k = 3)
    {
        var results = new List<(string, float)>();
        if (_lastDetections == null || _lastDetections.Count == 0)
            return results;
        foreach (var box in _lastDetections
                     .OrderByDescending(b => b.Confidence)
                     .Take(k))
        {
            results.Add((box.ClassName, box.Confidence));
        }
        return results;
    }
}
