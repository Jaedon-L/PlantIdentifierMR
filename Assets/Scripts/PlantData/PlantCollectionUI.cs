using System.Collections.Generic;
using UnityEngine;

public class PlantCollectionUI : MonoBehaviour
{
    [SerializeField] private RectTransform contentPanel; // assign Scroll View Content here
    [SerializeField] private GameObject itemPrefab;      // assign PlantListItem prefab

    private void Start()
    {
        // Subscribe to collection changes
        PlantCollectionManager.Instance.OnCollectionChanged += RefreshUI;
        // Initial population
        RefreshUI();
    }

    private void OnDestroy()
    {
        if (PlantCollectionManager.Instance != null)
            PlantCollectionManager.Instance.OnCollectionChanged -= RefreshUI;
    }

    public void RefreshUI()
    {
        // Clear existing children
        foreach (Transform child in contentPanel)
        {
            Destroy(child.gameObject);
        }

        // For each PlantEntry, instantiate a UI item
        IReadOnlyList<PlantEntry> entries = PlantCollectionManager.Instance.GetAllEntries();
        foreach (var entry in entries)
        {
            GameObject go = Instantiate(itemPrefab, contentPanel);
            PlantListItemController ctrl = go.GetComponent<PlantListItemController>();
            if (ctrl != null)
            {
                ctrl.Initialize(entry);
            }
        }
    }
}
