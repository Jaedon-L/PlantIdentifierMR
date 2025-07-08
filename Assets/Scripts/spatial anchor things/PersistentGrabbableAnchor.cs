using UnityEngine;
using Oculus.Interaction;

[RequireComponent(typeof(OVRSpatialAnchor))]
[RequireComponent(typeof(Grabbable))]
public class PersistentGrabbableAnchor : MonoBehaviour
{
    private OVRSpatialAnchor spatialAnchor;
    private Grabbable grabbable;

    void Awake()
    {
        spatialAnchor = GetComponent<OVRSpatialAnchor>();
        grabbable = GetComponent<Grabbable>();
        grabbable.WhenPointerEventRaised += OnGrabEvent;
    }

    private void OnGrabEvent(PointerEvent evt)
    {
        if (evt.Type == PointerEventType.Unselect)
        {
            spatialAnchor.Save((anchor, success) =>
            {
                Debug.Log(success ? "Anchor saved!" : "Failed to save anchor.");
            });
        }
    }

    void OnDestroy()
    {
        grabbable.WhenPointerEventRaised -= OnGrabEvent;
    }
}
