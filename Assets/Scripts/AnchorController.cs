using UnityEngine;

public class AnchorController : MonoBehaviour
{
    public int AnchorId { get; set; }
    public AnchorSpawner Spawner { get; set; }

    private void Awake()
    {
        // if nobody injected us yet, find the one-and-only spawner in the scene
        if (Spawner == null)
        {
            Spawner = FindFirstObjectByType<AnchorSpawner>();
            if (Spawner == null)
                Debug.LogError("AnchorController: no AnchorSpawner found in scene!");
        }
    }

    public void OnDeleteButtonClicked()
    {
        if (Spawner != null)
            Spawner.DeleteAnchorById(AnchorId);
        Destroy(gameObject);
    }
}
