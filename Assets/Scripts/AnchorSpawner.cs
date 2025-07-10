using Oculus.Interaction;
using UnityEngine;

public class AnchorSpawner : MonoBehaviour
{
    [SerializeField] private OVRHand _rightHand;
    [SerializeField] private OVRSkeleton _rightSkeleton;
    [SerializeField] private GameObject anchorPrefab;

    // PlayerPrefs keys
    private const string PrefKeyCount = "AnchorCount";
    private const string PrefKeyPos = "Anchor_Pos_{0}";

    private bool _rightPinchActive = false;

    // only allow one spawn when this is true
    private bool _canSpawnAnchor = false;

    // keep track of the single spawned anchor
    private GameObject _currentAnchor;

    void Start()
    {
        LoadSavedAnchors();
    }

    void Update()
    {
        if (_canSpawnAnchor)
            HandleRightPinch();
    }

    /// <summary>
    /// Call this from your UI “Spawn Anchor” button.
    /// Enables the next pinch to spawn exactly one anchor.
    /// </summary>
    public void EnableSpawnAnchor()
    {
        _canSpawnAnchor = true;
    }

    /// <summary>
    /// Call this from your UI “Delete Anchor” button.
    /// Deletes only the anchor that was spawned by EnableSpawnAnchor.
    /// </summary>
    public void DeleteCurrentAnchor()
    {
        if (_currentAnchor != null)
        {
            Destroy(_currentAnchor);
            _currentAnchor = null;
        }
    }

    /// <summary>
    /// Destroys every anchor in the scene (and clears saved prefs).
    /// </summary>
    [ContextMenu("Clear All Anchors")]
    public void ClearAllAnchors()
    {
        foreach (var existing in GameObject.FindGameObjectsWithTag("SpatialAnchor"))
            Destroy(existing);

        int saved = PlayerPrefs.GetInt(PrefKeyCount, 0);
        for (int i = 0; i < saved; i++)
            PlayerPrefs.DeleteKey(string.Format(PrefKeyPos, i));

        PlayerPrefs.DeleteKey(PrefKeyCount);
        PlayerPrefs.Save();

        _currentAnchor = null;
        _canSpawnAnchor = false;
    }

    private void HandleRightPinch()
    {
        bool isPinching = _rightHand.GetFingerIsPinching(OVRHand.HandFinger.Index);
        var confidence = _rightHand.GetFingerConfidence(OVRHand.HandFinger.Index);

        if (!_rightPinchActive && isPinching && confidence == OVRHand.TrackingConfidence.High)
        {
            _rightPinchActive = true;

            // Spawn one anchor, then disable further spawns
            SpawnAnchorAtIndexTip();
            _canSpawnAnchor = false;
        }
        else if (_rightPinchActive && !isPinching)
        {
            _rightPinchActive = false;
        }
    }
    public void DeleteAnchorById(int id)
    {
        // Remove the saved prefs for this anchor
        string posKey = string.Format(PrefKeyPos, id);
        if (PlayerPrefs.HasKey(posKey))
            PlayerPrefs.DeleteKey(posKey);

        // Now shift _all_ subsequent anchors down by one slot:
        int count = PlayerPrefs.GetInt(PrefKeyCount, 0);
        for (int i = id + 1; i < count; i++)
        {
            string src = string.Format(PrefKeyPos, i);
            string dst = string.Format(PrefKeyPos, i - 1);
            if (PlayerPrefs.HasKey(src))
            {
                string data = PlayerPrefs.GetString(src);
                PlayerPrefs.SetString(dst, data);
                PlayerPrefs.DeleteKey(src);
            }
        }

        // Decrement total count
        PlayerPrefs.SetInt(PrefKeyCount, count - 1);
        PlayerPrefs.Save();
    }

    private void SpawnAnchorAtIndexTip()
    {
        Transform tip = FindIndexTip(_rightSkeleton);
        if (tip == null)
        {
            Debug.LogWarning("Index tip not found, cannot spawn anchor.");
            return;
        }
        // Compute spawn position
        Vector3 spawnPos = tip.position;
        // Compute a rotation so that the anchor faces the camera on Y‑axis only:
        Vector3 toCam = Camera.main.transform.position - spawnPos;
        toCam.y = 0;                            // zero out vertical difference
        Quaternion faceUser = Quaternion.LookRotation(toCam, Vector3.up);
        // 1) Get the next available ID
        int newId = PlayerPrefs.GetInt(PrefKeyCount, 0);

        // 2) Instantiate your prefab at the tip position, upright
        GameObject go = Instantiate(anchorPrefab, tip.position, faceUser);
        go.tag = "SpatialAnchor";
        go.AddComponent<OVRSpatialAnchor>();
        // 3) Configure its AnchorController (which you have already placed on the prefab)
        var ctrl = go.GetComponent<AnchorController>();
        if (ctrl != null)
        {
            ctrl.AnchorId = newId;
            ctrl.Spawner = this;
        }
        else
        {
            Debug.LogError("SpawnAnchorAtIndexTip: prefab is missing an AnchorController component.");
        }

        // 4) Persist its position under that ID
        string posKey = string.Format(PrefKeyPos, newId);
        Vector3 pos = tip.position;
        PlayerPrefs.SetString(posKey, $"{pos.x},{pos.y},{pos.z}");

        // 5) Increment your saved count and write it out
        PlayerPrefs.SetInt(PrefKeyCount, newId + 1);
        PlayerPrefs.Save();

        // 6) Remember as “current” if you still need that
        _currentAnchor = go;

        Debug.Log($"Anchor #{newId} spawned at {pos} and saved.");
    }
    private void LoadSavedAnchors()
    {
        int count = PlayerPrefs.GetInt(PrefKeyCount, 0);
        for (int i = 0; i < count; i++)
        {
            string posKey = string.Format(PrefKeyPos, i);
            if (!PlayerPrefs.HasKey(posKey)) continue;

            var parts = PlayerPrefs.GetString(posKey).Split(',');
            if (parts.Length != 3) continue;

            Vector3 pos = new Vector3(
                float.Parse(parts[0]),
                float.Parse(parts[1]),
                float.Parse(parts[2])
            );

            GameObject anchor = Instantiate(anchorPrefab, pos, Quaternion.identity);
            anchor.tag = "SpatialAnchor";
            anchor.AddComponent<OVRSpatialAnchor>();
        }
    }

    private Transform FindIndexTip(OVRSkeleton skeleton)
    {
        if (skeleton.Bones == null) return null;
        foreach (var b in skeleton.Bones)
            if (b.Id == OVRSkeleton.BoneId.Hand_IndexTip)
                return b.Transform;
        return null;
    }
}
