using UnityEngine;
using DG.Tweening;
using UnityEngine.InputSystem;

public class ShowPanelOnSwipeForward : MonoBehaviour
{
    [SerializeField] private Vector3 correctScale = new Vector3(0.0054f, 0.0054f, 0.0054f);
    public OVRHand ovrHand;
    public GameObject panel;
    public Transform targetObject;
    //new
    [SerializeField] private float rayDistance = 0.2f; 

    void Start()
    {
        if (ovrHand == null)
        {
            GameObject rightHandObj = GameObject.Find("[BuildingBlock] Hand Tracking right");
            if (rightHandObj != null)
            {
                ovrHand = rightHandObj.GetComponent<OVRHand>();
            }

            if (ovrHand == null)
            {
                Debug.LogWarning("⚠️ OVRHand component not found on '[BuildingBlock] Hand Tracking right'.");
            }
        }
    }

    void Update()
    {
        HandleGesture();

        
        if (Keyboard.current.pKey.wasPressedThisFrame)
        {
            Debug.Log("⌨️ Keyboard P pressed - simulating gesture.");
            SimulateGesture();
        }
        HandleHandClick(); 
    }

    private void HandleGesture()
    {
        if (ovrHand == null || panel == null) return;

        var gesture = ovrHand.GetMicrogestureType();

        if (gesture == OVRHand.MicrogestureType.SwipeForward)
        {
            Debug.Log("➡️ SwipeForward detected.");
            SimulateGesture();
        }
    }

    private void SimulateGesture()
    {
        if (!panel.activeSelf)
        {
            
            panel.transform.position = targetObject.position + Vector3.up * 0.3f;
            panel.transform.rotation = Quaternion.identity;
            panel.SetActive(true);

            
            panel.transform.localScale = Vector3.zero;
            panel.transform.DOScale(correctScale, 0.5f).SetEase(Ease.OutBack);

            var cg = panel.GetComponent<CanvasGroup>();
            if (cg != null)
            {
                cg.alpha = 0;
                cg.DOFade(1, 0.5f);
            }

            
            var floating = targetObject.GetComponent<FloatingObject>();
            if (floating != null)
                floating.StopFloating();
        }
        else
        {
            
            HidePanel();
        }
    }

    private void HidePanel()
    {
        
        panel.transform.DOScale(Vector3.zero, 0.4f).SetEase(Ease.InBack);

        var cg = panel.GetComponent<CanvasGroup>();
        if (cg != null)
            cg.DOFade(0, 0.3f).OnComplete(() => panel.SetActive(false));
        else
            panel.SetActive(false);

        
        var floating = targetObject.GetComponent<FloatingObject>();
        if (floating != null)
            floating.ResumeFloating();
    }
    private void HandleHandClick()
    {
        if (ovrHand == null || !ovrHand.GetFingerIsPinching(OVRHand.HandFinger.Index)) return;

        Ray ray = new Ray(ovrHand.PointerPose.position, ovrHand.PointerPose.forward);
        if (Physics.Raycast(ray, out RaycastHit hit, rayDistance))
        {
            if (hit.transform == targetObject)
            {
                Debug.Log("🤚 Hand click on object detected!");
                SimulateGesture();
            }
        }
    }
    public void TogglePanelFromButton()
    {
        Debug.Log("🟢 Button clicked - toggling panel");
        SimulateGesture();
    }
}
