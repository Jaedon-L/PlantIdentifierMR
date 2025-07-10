using UnityEngine;

public class UIToggle : MonoBehaviour
{
    

    [ContextMenu("toggle")]
    public void OnButtonPress(GameObject target)
    {
        
        target.SetActive(!target.activeSelf);

        Debug.Log("pressed");
    }
}
