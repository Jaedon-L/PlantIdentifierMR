using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class IEInferenceTrigger : MonoBehaviour
{
    [SerializeField] private IEExecutor _ieExecutor;
    [SerializeField] private RawImage _outputImage;
    [SerializeField] private string _imageName;
    [SerializeField] private string _resourcesDirectory = "Assets/Resources";
    [SerializeField] private string _imagesDirectory = "Images";
    [SerializeField] private float _inferenceInterval = 5f;

    private List<string> _imageNames = new();
    private List<Texture2D> _imageTextures = new();
    private int _imageIndex = -1;
    private bool _isWaiting = false;

    private void Start()
    {
        LoadImageNames();
        Invoke(nameof(RunNextInference), 1f);
    }

    private void Update()
    {
        if (!_ieExecutor.IsRunning() && !_isWaiting)
        {
            _isWaiting = true;
            CancelInvoke(nameof(RunNextInference));
            Invoke(nameof(RunNextInference), _inferenceInterval);
        }
    }

    private void RunNextInference()
    {
        _imageIndex = (_imageIndex + 1) % _imageNames.Count;

        Texture2D texture = GetImageTexture(_imageIndex);
        _outputImage.texture = texture;
        _outputImage.SetNativeSize();
        _ieExecutor.RunInference(texture);
        _isWaiting = false;
    }

    private Texture2D GetImageTexture(int index)
    {
        if (index > _imageTextures.Count-1)
        {
            _imageTextures.Add(Resources.Load<Texture2D>($"{_imagesDirectory}/{_imageNames[index]}"));
        }
        return _imageTextures[index];
    }

    private void LoadImageNames()
    {
        // Search for JPG files
        string[] jpgFiles = System.IO.Directory.GetFiles($"{_resourcesDirectory}/{_imagesDirectory}", "*.jpg");
        foreach (string file in jpgFiles)
        {
            _imageNames.Add(System.IO.Path.GetFileNameWithoutExtension(file));
        }

        // Search for PNG files
        string[] pngFiles = System.IO.Directory.GetFiles($"{_resourcesDirectory}/{_imagesDirectory}", "*.png");
        foreach (string file in pngFiles)
        {
            _imageNames.Add(System.IO.Path.GetFileNameWithoutExtension(file));
        }

        Debug.Log($"Loaded {_imageNames.Count} images from {$"{_resourcesDirectory}/{_imagesDirectory}"}");
    }
}
