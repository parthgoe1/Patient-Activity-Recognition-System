using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class rotate : MonoBehaviour
{

    public Transform target;
    public GameObject allch;
    public int point=0;
    public List<GameObject> charec;
    // Start is called before the first frame update
    void Start()
    {

       // transform.LookAt(target);
      //// for(int i=0;i<2;i++)
      //  {
       //     allch.transform.GetChild(i).gameObject.SetActive(false);
       //     charec.Add(allch.transform.GetChild(i).gameObject);
       // }
      //  charec[point].SetActive(true);
       


    }

    // Update is called once per frame
    void Update()
    {
        point += 1;



        ///   charec[point].SetActive(true);

// ScreenCapture.CaptureScreenshot("D:/screenshots/SomeLevel" + point.ToString()+".png");
        transform.RotateAround(target.position, new Vector3(0f, 1.0f, 0.0f), 20 * Time.deltaTime * 2f);
// transform.Translate(Vector3.left * Time.deltaTime*2f);

    }
}
