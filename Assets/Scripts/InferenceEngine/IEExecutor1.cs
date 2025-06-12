using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;

public class IEExecutor1 : MonoBehaviour
{
    enum InferenceDownloadState
    {
        Running = 0,
        RequestingOutput = 1,
        Success = 2,
        Error = 3,
        Cleanup = 4,
        Completed = 5
    }

    [SerializeField] private int modelInputSize = 640;
    [SerializeField] private BackendType _backend = BackendType.CPU;
    [SerializeField] private ModelAsset _modelAsset;
    [SerializeField] private int _layersPerFrame = 25;
    [SerializeField] private float _confidenceThreshold = 0.25f;
    [SerializeField] private TextAsset _labelsAsset;
    [SerializeField] private Transform _displayLocation; // parent for boxes
    [SerializeField] private IEBoxer _ieBoxer;

    private Worker _worker;
    private IEnumerator _schedule;
    private InferenceDownloadState _state = InferenceDownloadState.Completed;
    private bool _started = false;
    private Tensor<float> _inputTensor;
    private Tensor<float> _outputTensor;
    private bool _waitingForReadback = false;
    private int _lastInputWidth, _lastInputHeight;
    private string[] _labels;

    void Start()
    {
        // Load model
        Model model = ModelLoader.Load(_modelAsset);
        _worker = new Worker(model, _backend);

        // Pre-allocate input tensor
        _inputTensor?.Dispose();
        _inputTensor = new Tensor<float>(new TensorShape(1, 3, modelInputSize, modelInputSize));

        // Warm-up
        Texture2D dummy = new Texture2D(modelInputSize, modelInputSize, TextureFormat.RGB24, false);
        Tensor<float> warm = new Tensor<float>(new TensorShape(1, 3, modelInputSize, modelInputSize));
        TextureConverter.ToTensor(dummy, warm, new TextureTransform());
        _worker.Schedule(warm);
        warm.Dispose();
        Object.Destroy(dummy);

        // Load labels once
        var lines = _labelsAsset.text.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
        _labels = System.Array.FindAll(lines, l => !string.IsNullOrWhiteSpace(l));
    }

    void Update()
    {
        UpdateInference();
    }

    public void RunInference(Texture inputTexture)
    {
        if (_started) return;
        if (inputTexture == null) return;

        // Convert to Texture2D
        Texture2D srcTex;
        if (inputTexture is Texture2D t2d) srcTex = t2d;
        else
        {
            RenderTexture rt = RenderTexture.GetTemporary(inputTexture.width, inputTexture.height, 0);
            Graphics.Blit(inputTexture, rt);
            RenderTexture.active = rt;
            srcTex = new Texture2D(inputTexture.width, inputTexture.height, TextureFormat.RGB24, false);
            srcTex.ReadPixels(new Rect(0, 0, inputTexture.width, inputTexture.height), 0, 0);
            srcTex.Apply();
            RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);
        }

        _lastInputWidth = srcTex.width;
        _lastInputHeight = srcTex.height;

        // Resize & pad
        Texture2D padded = ScaleAndPad(srcTex, modelInputSize);
        if (!ReferenceEquals(srcTex, inputTexture)) Object.Destroy(srcTex);

        // Fill tensor
        TextureConverter.ToTensor(padded, _inputTensor, new TextureTransform());
        Object.Destroy(padded);

        // Schedule
        _schedule = _worker.ScheduleIterable(_inputTensor);
        _state = InferenceDownloadState.Running;
        _started = true;
    }

    private void UpdateInference()
    {
        if (!_started) return;

        if (_state == InferenceDownloadState.Running)
        {
            int it = 0;
            while (_schedule.MoveNext())
            {
                if (++it % _layersPerFrame == 0) return;
            }
            _state = InferenceDownloadState.RequestingOutput;
        }
        else if (_state == InferenceDownloadState.RequestingOutput)
        {
            if (!_waitingForReadback)
            {
                _outputTensor = _worker.PeekOutput(0) as Tensor<float>;
                _outputTensor.ReadbackRequest();
                _waitingForReadback = true;
            }
            else if (_outputTensor.IsReadbackRequestDone())
            {
                var result = _outputTensor.ReadbackAndClone() as Tensor<float>;
                _waitingForReadback = false;
                _outputTensor.Dispose();
                _outputTensor = null;

                if (result != null && result.shape[0] > 0)
                {
                    ProcessDetections(result);
                    result.Dispose();
                    _state = InferenceDownloadState.Success;
                }
                else
                {
                    Debug.LogError("Output tensor empty or null");
                    result?.Dispose();
                    _state = InferenceDownloadState.Error;
                }
            }
        }
        else if (_state == InferenceDownloadState.Success || _state == InferenceDownloadState.Error)
        {
            _state = InferenceDownloadState.Cleanup;
        }
        else if (_state == InferenceDownloadState.Cleanup)
        {
            // Optionally: _inputTensor.Dispose(); // if not reusing
            _started = false;
            _state = InferenceDownloadState.Completed;
        }
    }

    private void ProcessDetections(Tensor<float> data)
    {
        int rows = data.shape[1];
        int numClasses = _labels.Length;
        var dets = new List<BoundingBox>();

        // Letterbox reverse parameters
        int origW = _lastInputWidth, origH = _lastInputHeight;
        int mSize = modelInputSize;
        float srcAR = (float)origW / origH;
        int resizedW, resizedH;
        if (srcAR > 1f) { resizedW = mSize; resizedH = Mathf.RoundToInt(mSize / srcAR); }
        else { resizedW = Mathf.RoundToInt(mSize * srcAR); resizedH = mSize; }
        float padX = (mSize - resizedW) / 2f;
        float padY = (mSize - resizedH) / 2f;
        float scaleX = (float)resizedW / origW;
        float scaleY = (float)resizedH / origH;

        for (int i = 0; i < rows; i++)
        {
            float cx = data[0, i, 0];
            float cy = data[0, i, 1];
            float w = data[0, i, 2];
            float h = data[0, i, 3];
            float rawObj = data[0, i, 4];
            float objConf = 1f / (1f + Mathf.Exp(-rawObj));
            if (objConf < _confidenceThreshold) continue;

            int bestC = -1;
            float bestLogit = float.NegativeInfinity;
            for (int c = 0; c < numClasses; c++)
            {
                float logit = data[0, i, 5 + c];
                if (logit > bestLogit) { bestLogit = logit; bestC = c; }
            }
            if (bestC < 0 || bestC >= numClasses) continue;

            // Softmax class probability
            float sumExp = 0f;
            for (int c = 0; c < numClasses; c++)
                sumExp += Mathf.Exp(data[0, i, 5 + c] - bestLogit);
            float classProb = 1f / sumExp;
            float finalScore = objConf * classProb;
            if (finalScore < _confidenceThreshold) continue;

            // Convert normalized to absolute padded coords
            float absCx = cx * mSize;
            float absCy = cy * mSize;
            float absW = w * mSize;
            float absH = h * mSize;
            // Remove padding, map to original
            float x0 = (absCx - absW/2f - padX) / scaleX;
            float y0 = (absCy - absH/2f - padY) / scaleY;
            float boxW = absW / scaleX;
            float boxH = absH / scaleY;
            if (boxW <= 0 || boxH <= 0) continue;
            x0 = Mathf.Clamp(x0, 0, origW);
            y0 = Mathf.Clamp(y0, 0, origH);
            boxW = Mathf.Clamp(boxW, 0, origW - x0);
            boxH = Mathf.Clamp(boxH, 0, origH - y0);

            BoundingBox bb = new BoundingBox
            {
                CenterX = x0 + boxW/2f,
                CenterY = y0 + boxH/2f,
                Width = boxW,
                Height = boxH,
                Label = _labels[bestC],
                ClassName = _labels[bestC],
                WorldPos = null,
                Score = finalScore
            };
            dets.Add(bb);
        }

        dets = ApplyNMS(dets, 0.45f);
        // Draw via IEBoxer
        if (_ieBoxer != null)
            _ieBoxer.DrawBoxes(dets, _lastInputWidth, _lastInputHeight);
        else
        {
            foreach (var d in dets)
                Debug.Log($"Detected {d.Label} @ ({d.CenterX - d.Width/2f:F1},{d.CenterY - d.Height/2f:F1},{d.Width:F1},{d.Height:F1}) conf {d.Score:F2}");
        }
    }

    private List<BoundingBox> ApplyNMS(List<BoundingBox> dets, float iouThresh)
    {
        var sorted = new List<BoundingBox>(dets);
        sorted.Sort((a, b) => b.Score.CompareTo(a.Score));
        var results = new List<BoundingBox>();
        while (sorted.Count > 0)
        {
            var best = sorted[0];
            results.Add(best);
            sorted.RemoveAt(0);
            sorted.RemoveAll(d =>
            {
                if (d.ClassName != best.ClassName) return false;
                var ra = new Rect(best.CenterX - best.Width/2f, best.CenterY - best.Height/2f, best.Width, best.Height);
                var rb = new Rect(d.CenterX - d.Width/2f, d.CenterY - d.Height/2f, d.Width, d.Height);
                return ComputeIoU(ra, rb) > iouThresh;
            });
        }
        return results;
    }

    private float ComputeIoU(Rect a, Rect b)
    {
        float x1 = Mathf.Max(a.xMin, b.xMin);
        float y1 = Mathf.Max(a.yMin, b.yMin);
        float x2 = Mathf.Min(a.xMax, b.xMax);
        float y2 = Mathf.Min(a.yMax, b.yMax);
        float w = Mathf.Max(0, x2 - x1);
        float h = Mathf.Max(0, y2 - y1);
        float inter = w * h;
        float union = a.width * a.height + b.width * b.height - inter;
        return union > 0 ? inter / union : 0f;
    }

    // ScaleAndPad same as before, plus FlipTextureVertically if needed
    Texture2D ScaleAndPad(Texture2D src, int targetSize)
    {
        float srcAR = (float)src.width / src.height;
        int resizedW, resizedH;
        if (srcAR > 1f) { resizedW = targetSize; resizedH = Mathf.RoundToInt(targetSize / srcAR); }
        else { resizedW = Mathf.RoundToInt(targetSize * srcAR); resizedH = targetSize; }

        RenderTexture rt = RenderTexture.GetTemporary(resizedW, resizedH);
        RenderTexture.active = rt;
        Graphics.Blit(src, rt);
        Texture2D resized = new Texture2D(resizedW, resizedH, TextureFormat.RGB24, false);
        resized.ReadPixels(new Rect(0, 0, resizedW, resizedH), 0, 0);
        resized.Apply();
        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        FlipTextureVertically(resized);

        Texture2D padded = new Texture2D(targetSize, targetSize, TextureFormat.RGB24, false);
        Color fill = new Color(0.5f, 0.5f, 0.5f);
        Color[] pix = new Color[targetSize * targetSize];
        for (int i = 0; i < pix.Length; i++) pix[i] = fill;
        padded.SetPixels(pix);
        int xOff = (targetSize - resizedW) / 2;
        int yOff = (targetSize - resizedH) / 2;
        padded.SetPixels(xOff, yOff, resizedW, resizedH, resized.GetPixels());
        padded.Apply();
        Object.Destroy(resized);
        return padded;
    }

    void FlipTextureVertically(Texture2D tex)
    {
        int w = tex.width, h = tex.height;
        Color[] pixels = tex.GetPixels();
        for (int y = 0; y < h/2; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int top = y * w + x;
                int bot = (h - y - 1) * w + x;
                Color tmp = pixels[top];
                pixels[top] = pixels[bot];
                pixels[bot] = tmp;
            }
        }
        tex.SetPixels(pixels);
        tex.Apply();
    }
}
