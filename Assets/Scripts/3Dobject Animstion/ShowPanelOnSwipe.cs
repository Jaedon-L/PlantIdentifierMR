using UnityEngine;
using DG.Tweening;
using UnityEngine.InputSystem;

public class ShowPanelOnSwipeForward : MonoBehaviour
{
    [SerializeField] private Vector3 correctScale = new Vector3(0.0054f, 0.0054f, 0.0054f);
    public OVRHand ovrHand;
    public GameObject panel;
    public Transform targetObject;

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

        // تست با کیبورد (فقط در Editor یا لپ‌تاپ)
        if (Keyboard.current.pKey.wasPressedThisFrame)
        {
            Debug.Log("⌨️ Keyboard P pressed - simulating gesture.");
            SimulateGesture();
        }
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
            // 🟢 اگر پنل خاموشه → ظاهرش کن
            panel.transform.position = targetObject.position + Vector3.up * 0.3f;
            panel.transform.rotation = Quaternion.identity;
            panel.SetActive(true);

            // Scale و Fade in
            panel.transform.localScale = Vector3.zero;
            panel.transform.DOScale(correctScale, 0.5f).SetEase(Ease.OutBack);

            var cg = panel.GetComponent<CanvasGroup>();
            if (cg != null)
            {
                cg.alpha = 0;
                cg.DOFade(1, 0.5f);
            }

            // متوقف کردن حرکت شناور
            var floating = targetObject.GetComponent<FloatingObject>();
            if (floating != null)
                floating.StopFloating();
        }
        else
        {
            // 🔴 اگر پنل روشنه → مخفیش کن
            HidePanel();
        }
    }

    private void HidePanel()
    {
        // انیمیشن Scale کوچیک + Fade out
        panel.transform.DOScale(Vector3.zero, 0.4f).SetEase(Ease.InBack);

        var cg = panel.GetComponent<CanvasGroup>();
        if (cg != null)
            cg.DOFade(0, 0.3f).OnComplete(() => panel.SetActive(false));
        else
            panel.SetActive(false);

        // ادامه دادن حرکت شناور
        var floating = targetObject.GetComponent<FloatingObject>();
        if (floating != null)
            floating.ResumeFloating();
    }
}
