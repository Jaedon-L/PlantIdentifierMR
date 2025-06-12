using System.Collections.Generic;
using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.UI;
using System;
using System.Linq;

public struct BoundingBox
{
    public float CenterX;
    public float CenterY;
    public float Width;
    public float Height;
    public string Label;
    public float Confidence;
    public Vector3? WorldPos;
    public string ClassName;

    public float Score { get; internal set; }
}

public class IEBoxer : MonoBehaviour
{
    [SerializeField] private Transform _displayLocation;
    [SerializeField] private TextAsset _labelsAsset;
    [SerializeField] private Color _boxColor;
    [SerializeField] private Sprite _boxTexture;
    [SerializeField] private Font _font;
    [SerializeField] private Color _fontColor;
    [SerializeField] private int _fontSize = 80;

    private string[] _labels;
    private List<GameObject> _boxPool = new();


    private void Start()
    {
        _labels = _labelsAsset.text.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);
        Debug.Log($"Loaded {_labels.Length} labels from {_labelsAsset.name}");
    }

    // public List<BoundingBox> DrawBoxes(Tensor<float> output, float imageWidth, float imageHeight)
    // {
    //     List<BoundingBox> boundingBoxes = new();

    //     var scaleX = imageWidth / 640;
    //     var scaleY = imageHeight / 640;

    //     var halfWidth = imageWidth / 2;
    //     var halfHeight = imageHeight / 2;

    //     int boxesFound = output.shape[1];
    //     if (boxesFound <= 0) return boundingBoxes;
    //     int numClasses = 46;

    //     var maxBoxes = Mathf.Min(boxesFound, 200);

    //     for (var n = 0; n < maxBoxes; n++)
    //     {
    //         float rawObj = output[0, n, 4];
    //         Debug.Log(rawObj);
    //         float objectness = 1f / (1f + Mathf.Exp(-rawObj)); 
    //         Debug.Log($"Raw centerX (before scaling): {output[0, n, 0]}, raw width: {output[0, n, 2]}");


    //         // Get bounding box center coordinates
    //         var centerX = output[0, n, 0] * scaleX;
    //         var centerY = output[0, n, 1] * scaleY;

    //         float maxScore = float.MinValue;
    //         int classId = -1;

    //         for (int c = 0; c < numClasses; c++)
    //         {
    //             float score = output[0, n, 5 + c];
    //             // Debug.Log($"Box {n}, Class {c}, Score: {output[0, n, 5 + c]}");
    //             if (score > maxScore)
    //             {
    //                 maxScore = score;
    //                 classId = c;
    //             }
    //         }

    //         // Apply numerically stable softmax
    //         float maxLogit = maxScore;
    //         float sumExp = 0f;
    //         for (int c = 0; c < numClasses; c++)
    //         {
    //             sumExp += Mathf.Exp(output[0, n, 5 + c] - maxLogit);
    //         }

    //         float classProb = Mathf.Exp(output[0, n, 5 + classId] - maxLogit) / sumExp;
    //         float confidence = objectness * classProb;

    //         // Filter low confidence
    //         if (float.IsNaN(confidence) || confidence < 0.5f) continue;
    //         // Get object class name
    //         var classname = _labels[classId];

    //         // Create a new bounding box
    //         var box = new BoundingBox
    //         {
    //             CenterX = centerX,
    //             CenterY = centerY,
    //             ClassName = classname,
    //             Width = output[0, n, 2] * scaleX,
    //             Height = output[0, n, 3] * scaleY,
    //             Label = $"{classname}",
    //         };

    //         Debug.Log($"Box {n}: {box.Label} - Center: ({box.CenterX}, {box.CenterY}), Size: ({box.Width}, {box.Height})");
    //         Debug.Log($"[CHECK] confidence: {confidence}");

    //         boundingBoxes.Add(box);

    //         DrawBox(box, n);
    //     }

    //     ClearBoxes(maxBoxes);

    //     return boundingBoxes;
    // }
    public List<BoundingBox> DrawBoxes(Tensor<float> output, float imageWidth, float imageHeight)
    {
        List<BoundingBox> finalBoxes = new List<BoundingBox>();

        var shape = output.shape;
        int rank = shape.rank; // expect 3
        if (rank != 3)
        {
            Debug.LogError($"Unexpected output.rank: {rank}. Expected 3.");
            return finalBoxes;
        }
        int batch = shape[0];
        int attrLen = shape[1];
        int totalPreds = shape[2];

        Debug.Log($"DrawBoxes: tensor.shape = [{batch},{attrLen},{totalPreds}]");
        if (batch != 1)
            Debug.LogWarning($"Batch size = {batch}. Usually 1 for inference.");

        int inferredNumClasses = attrLen - 5;
        if (inferredNumClasses <= 0)
        {
            Debug.LogError($"Cannot infer numClasses from attrLen={attrLen}");
            return finalBoxes;
        }
        Debug.Log($"Inferred numClasses = {inferredNumClasses}");
        if (_labels.Length != inferredNumClasses)
        {
            Debug.LogWarning($"Label count {_labels.Length} != inferredNumClasses {inferredNumClasses}. " +
                             "Ensure your label file matches the model.");
        }
        int numClasses = inferredNumClasses;

        // if (attrLen < 5 + 1)
        // {
        //     Debug.LogError($"attrLen={attrLen} too small to contain coords+obj");
        //     return finalBoxes;
        // }

        int inputSize = 640;
        int[] strides = new int[] { 8, 16, 32 };
        int expectedGridTotal = 0;

        foreach (int s in strides)
            expectedGridTotal += (inputSize / s) * (inputSize / s);

        if (totalPreds == expectedGridTotal)
        {
            Debug.Log($"Using raw-grid decoding branch (totalPreds={totalPreds})");
            // 3A. Raw grid outputs: decode with anchor-free YOLOv8 formulas

            List<BoundingBox> candidates = new List<BoundingBox>();
            int index = 0;
            int[] gridSizes = new int[strides.Length];
            for (int i = 0; i < strides.Length; i++)
                gridSizes[i] = inputSize / strides[i];  // {80,40,20}

            // For debugging: log a few raw values of the first prediction
            if (totalPreds > 0)
            {
                float sampleTx = output[0, 0, 0];
                float sampleTy = output[0, 1, 0];
                float sampleTw = output[0, 2, 0];
                float sampleTh = output[0, 3, 0];
                Debug.Log($"Sample raw[0]: tx={sampleTx:F3}, ty={sampleTy:F3}, tw={sampleTw:F3}, th={sampleTh:F3}");
            }
            //decode loops
            for (int s = 0; s < strides.Length; s++)
            {
                int stride = strides[s];
                int gridSize = gridSizes[s];

                for (int gy = 0; gy < gridSize; gy++)
                {
                    for (int gx = 0; gx < gridSize; gx++)
                    {
                        if (index >= totalPreds)
                        {
                            Debug.LogError($"Index {index} >= totalPreds {totalPreds}. Check iteration logic.");
                            break;
                        }

                        // Raw predictions
                        float tx = output[0, 0, index];
                        float ty = output[0, 1, index];
                        float tw = output[0, 2, index];
                        float th = output[0, 3, index];
                        float rawObj = output[0, 4, index];
                        if (index < 5)
                        {
                            Debug.Log($"[Debug] idx={index} rawObj={rawObj:F3} -> objectness={Sigmoid(rawObj):F3}");
                            // Also log a few class logits:
                            string classLogitsSample = "";
                            for (int c = 0; c < Mathf.Min(numClasses, 5); c++)
                            {
                                classLogitsSample += $"{output[0, 5 + c, index]:F3}, ";
                            }
                            Debug.Log($"[Debug] idx={index} class logits[0..4]={classLogitsSample}");
                        }

                        // Decode center on 640-scale
                        float sigmoidTx = Sigmoid(tx);
                        float sigmoidTy = Sigmoid(ty);
                        float bx = (sigmoidTx * 2f - 0.5f + gx) * stride;
                        float by = (sigmoidTy * 2f - 0.5f + gy) * stride;

                        // Decode width/height on 640-scale
                        float sigmoidTw = Sigmoid(tw);
                        float sigmoidTh = Sigmoid(th);
                        float pw = (sigmoidTw * 2f); pw = pw * pw * stride;  // (sigmoid(tw)*2)^2 * stride
                        float ph = (sigmoidTh * 2f); ph = ph * ph * stride;  // (sigmoid(th)*2)^2 * stride

                        // Normalize [0..1]
                        float cx_norm = bx / inputSize;
                        float cy_norm = by / inputSize;
                        float w_norm = pw / inputSize;
                        float h_norm = ph / inputSize;

                        float cx_pixel = cx_norm * imageWidth; // Scale to image resolution
                        float cy_pixel = cy_norm * imageHeight;
                        float w_pixel = w_norm * imageWidth;
                        float h_pixel = h_norm * imageHeight;

                        // Objectness
                        float objectness = Sigmoid(rawObj);

                        // Class probabilities: stable softmax
                        // float maxLogit = float.MinValue;
                        // int classId = -1;
                        // for (int c = 0; c < numClasses; c++)
                        // {
                        //     float logit = output[0, 5 + c, index];
                        //     if (logit > maxLogit) { maxLogit = logit; classId = c; }
                        // }
                        // //softmax
                        // float sumExp = 0f;
                        // for (int c = 0; c < numClasses; c++)
                        //     sumExp += Mathf.Exp(output[0, 5 + c, index] - maxLogit);

                        // float classProb = Mathf.Exp(output[0, 5 + classId, index] - maxLogit) / sumExp;

                        // float confidence = objectness * classProb;
                        //  end softmax
                        // New sigmoid-based classification:
                        float maxScore = float.MinValue;
                        int classId = -1;
                        // Find class with highest sigmoid(logit)
                        for (int c = 0; c < numClasses; c++)
                        {
                            float logit = output[0, 5 + c, index];
                            float score = Sigmoid(logit);
                            if (score > maxScore)
                            {
                                maxScore = score;
                                classId = c;
                            }
                        }
                        float classScore = maxScore;           // sigmoid(logit) of chosen class
                        float confidence = objectness * classScore;


                        if (index < 5)
                        {
                            Debug.Log($"Pred {index}: obj={objectness:F3}, classProb={classScore:F3}, conf={confidence:F3}, classId={classId}");
                        }

                        // Confidence threshold
                        const float confThreshold = 0.3f;
                        if (float.IsNaN(confidence) || confidence < confThreshold)
                        {
                            Debug.Log($"Skipping idx={index}, conf={confidence:F3} < {confThreshold}");
                        }
                        else if (classId < 0 || classId >= _labels.Length)
                        {
                            Debug.LogWarning($"Skipping idx={index}: invalid classId={classId}, labels.Length={_labels.Length}");
                        }
                        else
                        {
                            var box = new BoundingBox
                            {
                                CenterX = cx_pixel,
                                CenterY = cy_pixel,
                                Width = w_pixel,
                                Height = h_pixel,
                                ClassName = _labels[classId],
                                Confidence = confidence,
                                Label = $"{_labels[classId]}: {confidence:F2}"
                            };
                            candidates.Add(box);
                        }
                        index++;
                    }
                }
            }
            if (index != totalPreds)
                Debug.LogWarning($"Decoded {index} but expected {totalPreds}. Check consistency.");

            // NMS per class
            float iouThreshold = 0.55f;

            foreach (var group in candidates.GroupBy(b => b.ClassName))
            {
                var list = group.OrderByDescending(b => b.Confidence).ToList();
                while (list.Count > 0)
                {
                    var best = list[0];
                    finalBoxes.Add(best);
                    list.RemoveAt(0);
                    list = list.Where(b => IoU(b, best) < iouThreshold).ToList();
                }
            }
            Debug.Log($"After NMS: {finalBoxes.Count} boxes");

            // Optionally keep only top-K by confidence:
            finalBoxes = finalBoxes.OrderByDescending(b => b.Confidence).Take(5).ToList();
        }
        else
        {
            Debug.Log($"Using post-processed branch (totalPreds={totalPreds}, attrLen={attrLen})");
            for (int n = 0; n < totalPreds; n++)
            {
                if (attrLen >= 6)
                {
                    if (n >= totalPreds) break;

                    float rawCx = output[0, 0, n];
                    float rawCy = output[0, 1, n];
                    float rawW = output[0, 2, n];
                    float rawH = output[0, 3, n];
                    float confidence = output[0, 4, n];
                    int classId = Mathf.RoundToInt(output[0, 5, n]);

                    const float confThreshold = 0.25f;
                    if (confidence < confThreshold) continue;

                    float cx_pixel = (rawCx / 640f) * imageWidth;
                    float cy_pixel = (rawCy / 640f) * imageHeight;
                    float w_pixel = (rawW / 640f) * imageWidth;
                    float h_pixel = (rawH / 640f) * imageHeight;

                    // if (classId < 0 || classId >= _labels.Length) { Debug.LogWarning($"Invalid classId {classId} at n={n}"); continue; }
                    if (classId < 0 || classId >= _labels.Length)
                    {
                        Debug.LogWarning($"Invalid classId={classId} at prediction {n}. Labels length={_labels.Length}. Possible mismatch between model classes and label file.");
                        continue;
                    }
                    var box = new BoundingBox
                    {
                        CenterX = cx_pixel,
                        CenterY = cy_pixel,
                        Width = w_pixel,
                        Height = h_pixel,
                        ClassName = _labels[classId],
                        Confidence = confidence,
                        Label = $"{_labels[classId]}: {confidence:F2}"
                    };
                    finalBoxes.Add(box);
                    // DrawBox(box, finalBoxes.Count - 1);
                }
                else
                {
                    // If attrLen differs, log some values to inspect format:
                    string sample = "";
                    int toLog = Mathf.Min(attrLen, 8);
                    for (int k = 0; k < toLog; k++)
                        sample += output[0, k, n].ToString("F3") + ",";
                    Debug.LogError($"Unexpected attrLen={attrLen}. Sample vals for n={n}: [{sample.TrimEnd(',')}]");
                    break;
                }
            }
        }

        // 4. Draw finalBoxes (for raw-grid branch, drawing done after NMS; for post-processed branch, some drawn above)
        // If raw-grid branch: need to draw here:
        RectTransform imageRect = _displayLocation.GetComponent<RectTransform>();
        float uiWidth = imageRect.rect.width;
        float uiHeight = imageRect.rect.height;

        Debug.Log($"Drawing {finalBoxes.Count} boxes");
        for (int i = 0; i < finalBoxes.Count; i++)
        {
            var box = finalBoxes[i];
            // float xMin = Mathf.Clamp(box.CenterX - box.Width / 2f, 0f, imageWidth);
            // float yMin = Mathf.Clamp(box.CenterY - box.Height / 2f, 0f, imageHeight);
            // float xMin = box.CenterX - box.Width / 2f;
            // float yMin = box.CenterY - box.Height / 2f;

            // // Clamp so boxes don’t go off-screen
            // xMin = Mathf.Clamp(xMin, 0f, imageWidth - box.Width);
            // yMin = Mathf.Clamp(yMin, 0f, imageHeight - box.Height);

            // // If Canvas origin is top-left, adjust y: float yPos = imageHeight - yMin - box.Height; pos=(xMin,yPos)
            // Vector3 pos = new Vector3(xMin, yMin, 0f);
            // normalized center:
            float cx_norm = box.CenterX / imageWidth;
            float cy_norm = box.CenterY / imageHeight;
            float w_norm = box.Width / imageWidth;
            float h_norm = box.Height / imageHeight;

            // UI sizes:
            float uiBoxW = w_norm * uiWidth;
            float uiBoxH = h_norm * uiHeight;

            // UI center offset from image center (pivot 0.5,0.5):
            float uiCenterX = (cx_norm - 0.5f) * uiWidth;
            float uiCenterY = (cy_norm - 0.5f) * uiHeight;

            GameObject panel;
            if (i < _boxPool.Count && _boxPool[i] != null)
            {
                panel = _boxPool[i];
                panel.SetActive(true);
            }
            else
            {
                panel = CreateNewBox(_boxColor);
            }
            // panel.transform.localPosition = pos;

            RectTransform rt = panel.GetComponent<RectTransform>();
            rt.SetParent(_displayLocation, false);
            rt.sizeDelta = new Vector2(uiBoxW, uiBoxH);
            rt.anchoredPosition = new Vector2(uiCenterX, uiCenterY);
            // rt.sizeDelta = new Vector2(box.Width, box.Height);
            var labelComp = panel.GetComponentInChildren<Text>();
            labelComp.text = box.Label;
        }

        // 5. Clear leftover panels
        ClearBoxes(finalBoxes.Count);

        return finalBoxes;
    }

    // Helper: sigmoid
    private float Sigmoid(float x)
    {
        return 1f / (1f + Mathf.Exp(-x));
    }

    // Helper: Intersection-over-Union
    private float IoU(BoundingBox a, BoundingBox b)
    {
        float axMin = a.CenterX - a.Width / 2f, axMax = a.CenterX + a.Width / 2f;
        float ayMin = a.CenterY - a.Height / 2f, ayMax = a.CenterY + a.Height / 2f;
        float bxMin = b.CenterX - b.Width / 2f, bxMax = b.CenterX + b.Width / 2f;
        float byMin = b.CenterY - b.Height / 2f, byMax = b.CenterY + b.Height / 2f;

        float interXMin = Mathf.Max(axMin, bxMin);
        float interYMin = Mathf.Max(ayMin, byMin);
        float interXMax = Mathf.Min(axMax, bxMax);
        float interYMax = Mathf.Min(ayMax, byMax);
        float interW = Mathf.Max(0f, interXMax - interXMin);
        float interH = Mathf.Max(0f, interYMax - interYMin);
        float interArea = interW * interH;
        float areaA = a.Width * a.Height;
        float areaB = b.Width * b.Height;
        // return interArea / (areaA + areaB - interArea);
        return areaA + areaB - interArea > 0f ? interArea / (areaA + areaB - interArea) : 0f;
    }
    public void ClearAllBoxes()
{
    for (int i = 0; i < _boxPool.Count; i++)
    {
        if (_boxPool[i] != null)
            _boxPool[i].SetActive(false);
    }
}
    public void ClearBoxes(int lastBoxCount)
    {
        if (lastBoxCount < _boxPool.Count)
        {
            for (int i = lastBoxCount; i < _boxPool.Count; i++)
            {
                if (_boxPool[i] != null)
                {
                    _boxPool[i].SetActive(false);
                }
            }
        }
    }

    private void DrawBox(BoundingBox box, int id)
    {
        GameObject panel;
        if (id < _boxPool.Count)
        {
            panel = _boxPool[id];
            if (panel == null)
            {
                panel = CreateNewBox(_boxColor);
            }
            else
            {
                panel.SetActive(true);
            }
        }
        else
        {
            panel = CreateNewBox(_boxColor);
        }

        // Set box position
        panel.transform.localPosition = new Vector3(box.CenterX, -box.CenterY, box.WorldPos.HasValue ? box.WorldPos.Value.z : 0.0f);

        // Set box size
        RectTransform rectTransform = panel.GetComponent<RectTransform>();
        rectTransform.sizeDelta = new Vector2(box.Width, box.Height);

        // Set label text
        Text label = panel.GetComponentInChildren<Text>();
        label.text = box.Label;
    }

    private GameObject CreateNewBox(Color color)
    {
        // Create the box and set image
        GameObject panel = new("ObjectBox");
        panel.AddComponent<CanvasRenderer>();

        Image image = panel.AddComponent<Image>();
        image.color = color;
        image.sprite = _boxTexture;
        image.type = Image.Type.Sliced;
        image.fillCenter = false;
        panel.transform.SetParent(_displayLocation, false);

        // Create the label
        GameObject textGameObject = new("ObjectLabel");
        textGameObject.AddComponent<CanvasRenderer>();
        textGameObject.transform.SetParent(panel.transform, false);

        Text text = textGameObject.AddComponent<Text>();
        text.font = _font;
        text.color = _fontColor;
        text.fontSize = _fontSize;
        text.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rectTransform = textGameObject.GetComponent<RectTransform>();
        rectTransform.offsetMin = new Vector2(20, rectTransform.offsetMin.y);
        rectTransform.offsetMax = new Vector2(0, rectTransform.offsetMax.y);
        rectTransform.offsetMin = new Vector2(rectTransform.offsetMin.x, 0);
        rectTransform.offsetMax = new Vector2(rectTransform.offsetMax.x, 30);
        rectTransform.anchorMin = new Vector2(0, 0);
        rectTransform.anchorMax = new Vector2(1, 1);

        _boxPool.Add(panel);

        return panel;
    }

    internal void DrawBoxes(List<BoundingBox> dets, int lastInputWidth, int lastInputHeight)
    {
        throw new NotImplementedException();
    }
}
