using UnityEngine;

public class TogleMenu : MonoBehaviour
{
    [SerializeField] private GameObject levelPanel;  // اشاره به پنل Level

    private bool isVisible = false;

    public void OnToggleLevelClicked()
    {
        isVisible = !isVisible;
        levelPanel.SetActive(isVisible);
    }
}
