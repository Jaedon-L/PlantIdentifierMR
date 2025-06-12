using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using PassthroughCameraSamples;
using TMPro;

public class IEPassthroughTrigger : MonoBehaviour
{
    [SerializeField] private WebCamTextureManager _webCamTextureManager;
    [SerializeField] private RawImage _outputImage;
    [SerializeField] private IEExecutor _ieExecutor;
    [SerializeField] private IEBoxer _ieBoxer;
    [SerializeField] private Canvas scanCanvas;
    private bool _scanRequested;
    [SerializeField] private GameObject scanButton;
    [SerializeField] private Canvas results;
    [SerializeField] private TextMeshPro resultsText;
    private bool _scanInProgress = false;
    [SerializeField] private GameObject YesNoMenu;

    private bool _loggedMissingReferences = false;

    private IEnumerator Start()
    {
        // Check required references once
        CheckRequiredReferences();

        // Wait until Sentis model is loaded
        if (_ieExecutor == null)
        {
            yield break;
        }
        while (!_ieExecutor.IsModelLoaded)
            yield return null;

        Debug.Log("IEPassthroughTrigger: Sentis model is loaded");

        // Initialize UI state
        if (scanCanvas != null)
            scanCanvas.enabled = false;

        if (resultsText != null)
            resultsText.text = "";

        if (results != null)
            results.enabled = false;

        if (YesNoMenu != null)
            YesNoMenu.SetActive(false);

        // Make sure scanButton is active/interactable
        if (scanButton != null)
            scanButton.SetActive(true);
    }

    private void Update()
    {
        // Guard: if missing key references, skip update logic
        if (_ieExecutor == null || _webCamTextureManager == null)
            return;

        // Only start inference when requested
        if (_scanRequested && !_scanInProgress && !_ieExecutor.IsRunning())
        {
            var camTex = _webCamTextureManager.WebCamTexture;
            if (camTex == null)
            {
                Debug.LogWarning("IEPassthroughTrigger: WebCamTexture is null; cannot run inference.");
                _scanRequested = false;
                _scanInProgress = false;
                // Re-enable scanButton so user can try again
                if (scanButton != null)
                    scanButton.SetActive(true);
                return;
            }

            _scanRequested = false;
            _scanInProgress = true;

            // Show camera feed in RawImage if assigned
            if (_outputImage != null)
            {
                _outputImage.texture = camTex;
                _outputImage.SetNativeSize();
            }
            else
            {
                Debug.LogWarning("IEPassthroughTrigger: _outputImage is not assigned.");
            }

            // Clear previous boxes if IEBoxer assigned
            if (_ieBoxer != null)
            {
                _ieBoxer.ClearAllBoxes();
            }

            // Run inference
            _ieExecutor.RunInference(camTex);

            // After scheduling inference, wait for completion
            StartCoroutine(HandlePostScan());
        }
    }

    /// <summary>
    /// Called when user taps Scan button.
    /// </summary>
    public void OnScanButtonClicked()
    {
        StartCoroutine(StartCountdownBeforeScan(3));
        // Check references
        if (_ieExecutor == null)
        {
            Debug.LogError("IEPassthroughTrigger: IEExecutor is not assigned.");
            return;
        }
        if (_webCamTextureManager == null)
        {
            Debug.LogError("IEPassthroughTrigger: WebCamTextureManager is not assigned.");
            return;
        }
        // Prevent double-click if already scanning
        if (_ieExecutor.IsRunning() || _scanInProgress)
            return;

        // Clear any drawn boxes before new scan
        if (_ieBoxer != null)
        {
            _ieBoxer.ClearAllBoxes();
        }

        // Enable results canvas so we can show countdown/result
        if (results != null)
            results.enabled = true;
        else
            Debug.LogWarning("IEPassthroughTrigger: results Canvas is not assigned.");

        // Clear any previous text
        if (resultsText != null)
            resultsText.text = "";
        else
            Debug.LogWarning("IEPassthroughTrigger: resultsText (TextMeshProUGUI) is not assigned.");

        Debug.Log("Scan button clicked, starting countdown.");

    }

    private IEnumerator StartCountdownBeforeScan(int seconds)
    {

        // Disable scanButton immediately
        if (scanButton != null)
            scanButton.SetActive(false);
        else
            Debug.LogWarning("IEPassthroughTrigger: scanButton GameObject is not assigned.");
        for (int i = seconds; i > 0; i--)
        {
            if (resultsText != null)
                resultsText.text = i.ToString();
            yield return new WaitForSeconds(1f);
        }
        // Countdown complete
        if (resultsText != null)
            resultsText.text = "Scanning...";

        // Trigger the actual inference in Update()
        _scanRequested = true;

        // Show scanCanvas overlay if assigned
        if (scanCanvas != null)
            scanCanvas.enabled = true;
        else
            Debug.LogWarning("IEPassthroughTrigger: scanCanvas is not assigned.");

    }

    private IEnumerator HandlePostScan()
    {
        // Wait until inference is actually done
        if (_ieExecutor == null)
            yield break;

        while (_ieExecutor.IsRunning())
            yield return null;

        // Attempt to get the top prediction
        string top = null;
        try
        {
            top = _ieExecutor.GetTopPrediction();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"IEPassthroughTrigger: Exception in GetTopPrediction(): {e}");
        }

        // If no valid prediction or explicitly “No detections”, show that and skip Yes/No menu
        if (string.IsNullOrEmpty(top) || top.Equals("No detections", System.StringComparison.OrdinalIgnoreCase))
        {
            if (resultsText != null)
                resultsText.text = "No detections";
            else
                Debug.LogWarning("IEPassthroughTrigger: resultsText is null; cannot display 'No detections'.");

            // Ensure Yes/No menu is hidden
            if (YesNoMenu != null)
                YesNoMenu.SetActive(false);

            // Re-enable the scan button so user can try again
            if (scanButton != null)
                scanButton.SetActive(true);
            else
                Debug.LogWarning("IEPassthroughTrigger: scanButton is not assigned.");

            // Optionally hide scanCanvas overlay if still active
            if (scanCanvas != null)
                scanCanvas.enabled = false;
                
            if (results != null)
                results.enabled = false;

            _scanInProgress = false;
            yield break;
        }

        // Otherwise, we have a valid detection: display it
        if (resultsText != null)
            resultsText.text = $"Detected: {top}";
        else
            Debug.LogWarning("IEPassthroughTrigger: resultsText is null; cannot display detected result.");

        // Show Yes/No menu for user to confirm adding
        if (YesNoMenu != null)
            YesNoMenu.SetActive(true);
        else
            Debug.LogWarning("IEPassthroughTrigger: YesNoMenu GameObject not assigned; cannot show confirmation UI.");

        // Hide scanCanvas overlay if you used it for “scanning...” overlay
        if (scanCanvas != null)
            scanCanvas.enabled = false;

        _scanInProgress = false;
        Debug.Log("Scan complete, showing Yes/No menu.");
    }
    /// <summary>
    /// Called when user taps “Yes” to add to collection.
    /// </summary>
    public void OnYesButtonClicked()
    {
        // Get predicted class
        string classOnly = null;
        try
        {
            if (_ieExecutor != null)
                classOnly = _ieExecutor.GetTopPrediction();
        }
        catch (System.Exception e)
        {
            Debug.LogError($"IEPassthroughTrigger: Exception in GetTopPrediction(): {e}");
        }

        if (string.IsNullOrEmpty(classOnly) || classOnly == "No detections")
        {
            if (resultsText != null)
                resultsText.text = "Nothing to add.";
            else
                Debug.LogWarning("IEPassthroughTrigger: resultsText is null; cannot show 'Nothing to add.'");
            if (YesNoMenu != null)
                YesNoMenu.SetActive(false);
            return;
        }

        // Add to collection via singleton
        if (PlantCollectionManager.Instance != null)
        {
            PlantCollectionManager.Instance.AddEntry(classOnly);
            if (resultsText != null)
                resultsText.text = $"Added: {classOnly}";
        }
        else
        {
            Debug.LogError("IEPassthroughTrigger: PlantCollectionManager.Instance is null. Ensure PlantCollectionManager exists in scene.");
            if (resultsText != null)
                resultsText.text = "Error: cannot add.";
        }

        // Hide Yes/No menu
        if (YesNoMenu != null)
            YesNoMenu.SetActive(false);

        // Re-enable scan button
        if (scanButton != null)
            scanButton.SetActive(true);

        if (results != null)
            results.enabled = false;
    }

    /// <summary>
    /// Called when user taps “No” to discard.
    /// </summary>
    public void OnNoButtonClicked()
    {
        if (resultsText != null)
            resultsText.text = "Discarded.";
        else
            Debug.LogWarning("IEPassthroughTrigger: resultsText is null; cannot show 'Discarded.'");

        // Hide Yes/No menu
        if (YesNoMenu != null)
            YesNoMenu.SetActive(false);

        // Re-enable scan button
        if (scanButton != null)
            scanButton.SetActive(true);

        if (results != null)
            results.enabled = false;

        // Optionally hide scanCanvas overlay:
        // if (scanCanvas != null) scanCanvas.enabled = false;
    }

    /// <summary>
    /// Check once for missing required references; logs errors if any are null.
    /// </summary>
    private void CheckRequiredReferences()
    {
        if (_loggedMissingReferences)
            return;
        _loggedMissingReferences = true;

        if (_webCamTextureManager == null)
            Debug.LogError("IEPassthroughTrigger: WebCamTextureManager is not assigned in Inspector.");
        if (_outputImage == null)
            Debug.LogError("IEPassthroughTrigger: RawImage (_outputImage) is not assigned in Inspector.");
        if (_ieExecutor == null)
            Debug.LogError("IEPassthroughTrigger: IEExecutor is not assigned in Inspector.");
        if (_ieBoxer == null)
            Debug.LogWarning("IEPassthroughTrigger: IEBoxer is not assigned in Inspector; boxes cannot be cleared/drawn.");
        if (scanCanvas == null)
            Debug.LogWarning("IEPassthroughTrigger: scanCanvas is not assigned in Inspector.");
        if (scanButton == null)
            Debug.LogWarning("IEPassthroughTrigger: scanButton GameObject is not assigned in Inspector.");
        if (results == null)
            Debug.LogWarning("IEPassthroughTrigger: results Canvas is not assigned in Inspector.");
        if (resultsText == null)
            Debug.LogWarning("IEPassthroughTrigger: resultsText TextMeshProUGUI is not assigned in Inspector.");
        if (YesNoMenu == null)
            Debug.LogWarning("IEPassthroughTrigger: YesNoMenu GameObject is not assigned in Inspector.");
    }
}
