using UnityEngine;
using UnityEngine.UI;

public class ClosePanelOnToggle : MonoBehaviour
{
    [SerializeField] private Button buttonToClose;      
    [SerializeField] private GameObject panelToClose;

    private void Start()
    {
        if (buttonToClose != null)
            buttonToClose.onClick.AddListener(OnButtonClicked);
    }

    private void OnButtonClicked()
    {
        if (panelToClose != null)
            panelToClose.SetActive(false);
    }
}
