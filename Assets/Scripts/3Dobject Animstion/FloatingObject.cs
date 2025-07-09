using UnityEngine;
using DG.Tweening;

public class FloatingObject : MonoBehaviour
{
    private Tween floatTween;
    public AudioSource audioSource; 
    public ParticleSystem particles; 
                                     
    private float baseY; 
    void Start()
    {
        baseY = transform.position.y; 
        StartFloating();
    }

    public void StartFloating()
    {
        transform.position = new Vector3(transform.position.x, baseY, transform.position.z);
        floatTween = transform.DOMoveY(transform.position.y + 0.2f, 1f)
                              .SetLoops(-1, LoopType.Yoyo)
                              .SetEase(Ease.InOutSine);

        
        if (audioSource != null && !audioSource.isPlaying)
            audioSource.Play();

        if (particles != null && !particles.isPlaying)
            particles.Play();
    }

    public void StopFloating()
    {
        if (floatTween != null && floatTween.IsActive())
            floatTween.Kill();

        if (audioSource != null)
            audioSource.Stop();

        if (particles != null)
            particles.Stop();
    }

    public void ResumeFloating()
    {
        StopFloating(); 
        StartFloating();
    }
}
