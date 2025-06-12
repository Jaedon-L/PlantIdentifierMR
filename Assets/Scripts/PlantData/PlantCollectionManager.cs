using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class PlantCollectionManager : MonoBehaviour
{
    public static PlantCollectionManager Instance { get; private set; }

    private string _dataFilePath;
    private List<PlantEntry> _entries = new List<PlantEntry>();

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);

        // Determine JSON file path
        _dataFilePath = Path.Combine(Application.persistentDataPath, "plant_collection.json");
        LoadFromDisk();
    }

    /// <summary>
    /// Returns a read-only copy of current entries.
    /// </summary>
    public IReadOnlyList<PlantEntry> GetAllEntries()
    {
        return _entries;
    }

    /// <summary>
    /// Add a new plant entry, save immediately, and notify listeners.
    /// </summary>
    public void AddEntry(string plantName, string thumbnailFileName = null)
    {
        string ts = System.DateTime.UtcNow.ToString("o");
        PlantEntry entry = new PlantEntry(plantName, ts, thumbnailFileName);
        _entries.Add(entry);
        SaveToDisk();
        // Optionally fire an event or callback so UI can refresh
        OnCollectionChanged?.Invoke();
    }

    /// <summary>
    /// Save current _entries to JSON file.
    /// </summary>
    private void SaveToDisk()
    {
        try
        {
            PlantCollectionData wrapper = new PlantCollectionData
            {
                entries = _entries.ToArray()
            };
            string json = JsonUtility.ToJson(wrapper, prettyPrint: true);
            File.WriteAllText(_dataFilePath, json);
            Debug.Log($"PlantCollectionManager: Saved {_entries.Count} entries to {_dataFilePath}");
        }
        catch (System.Exception e)
        {
            Debug.LogError($"PlantCollectionManager: Failed to save to disk: {e}");
        }
    }

    /// <summary>
    /// Load from JSON file if exists; otherwise start with empty list.
    /// </summary>
    private void LoadFromDisk()
    {
        if (File.Exists(_dataFilePath))
        {
            try
            {
                string json = File.ReadAllText(_dataFilePath);
                PlantCollectionData wrapper = JsonUtility.FromJson<PlantCollectionData>(json);
                if (wrapper != null && wrapper.entries != null)
                {
                    _entries = new List<PlantEntry>(wrapper.entries);
                    Debug.Log($"PlantCollectionManager: Loaded {_entries.Count} entries from {_dataFilePath}");
                }
                else
                {
                    _entries = new List<PlantEntry>();
                    Debug.Log("PlantCollectionManager: No entries in JSON, starting empty.");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"PlantCollectionManager: Failed to load from disk: {e}");
                _entries = new List<PlantEntry>();
            }
        }
        else
        {
            _entries = new List<PlantEntry>();
            Debug.Log("PlantCollectionManager: No existing data file, starting empty.");
        }
    }

    /// <summary>
    /// Optional: Clear all entries (and delete file).
    /// </summary>
    public void ClearAll()
    {
        _entries.Clear();
        if (File.Exists(_dataFilePath))
        {
            File.Delete(_dataFilePath);
        }
        OnCollectionChanged?.Invoke();
    }

    /// <summary>
    /// Event fired whenever the collection changes (entry added or cleared).
    /// UI can subscribe to refresh display.
    /// </summary>
    public event System.Action OnCollectionChanged;
}
