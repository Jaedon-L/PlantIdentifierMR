using UnityEngine;
using Meta.XR.BuildingBlocks;
using Meta.XR.MRUtilityKit;
using UnityEngine.Events;

public class PlantAnchorPlacer : MonoBehaviour
{
    [SerializeField] private SpatialAnchorSpawnerBuildingBlock anchorSpawner;
    [SerializeField] private SpatialAnchorLoaderBuildingBlock anchorLoader;

    void Awake()
    {
        anchorLoader.LoadAnchorsFromDefaultLocalStorage();
    }

    // این تابع را به دکمه وصل می‌کنیم
    public void PlaceAnchorOnPlant()
    {
        foreach (var room in MRUK.Instance.Rooms)
        {
            foreach (var anchor in room.Anchors)
            {
                if (anchor.Label.HasFlag(MRUKAnchor.SceneLabels.PLANT))
                {
                    Vector3 pos = anchor.transform.position;
                    Quaternion rot = anchor.transform.rotation;

                    anchorSpawner.SpawnSpatialAnchor(pos, rot);
                    Debug.Log("✅ Anchor placed on plant.");
                    return;
                }
            }
        }

        Debug.LogWarning("⚠️ No plant anchor found.");
    }
}
