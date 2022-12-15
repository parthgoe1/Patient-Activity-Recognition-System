using System;
using System.IO;
using UnityEngine;

public class CameraRotator : MonoBehaviour
{
    private float speed = 360f;
    [SerializeField]
    private GameObject target,current_object;
    private int nChild;
    private int i,j,num_of_images=0;
    private bool counter;
    private int frame_count = 0;
    private String rootFolder = "D:/screenshots/";
    private String folderName;
    
    void Start()
    {
        nChild = target.transform.childCount;
        // Debug.Log(nChild);
        i = 0;
        j = 0;
        
      


    }
    // Update is called once per frame
    void Update()
    {

        current_object.transform.position= target.transform.GetChild(j).transform.position;

        folderName = rootFolder + current_object.transform.name+'/';
        if (!Directory.Exists(folderName))
        {
            Directory.CreateDirectory(folderName);
        }
        

        if (num_of_images!=200)
        {
            ScreenCapture.CaptureScreenshot(folderName + i.ToString() + ".png");
            i += 1;
            num_of_images += 1;
        }
        if(num_of_images==200)
        {
            num_of_images =0;
            j += 1;
            
        }









        /*
        if (transform.eulerAngles.y < 350f)
        {
            target.transform.GetChild(i).gameObject.SetActive(true);
            // If directory does not exist, create it

            folderName = rootFolder + target.transform.GetChild(i).name + "/_";
            if (!Directory.Exists(folderName))
            {
                Directory.CreateDirectory(folderName);
            }

            frame_count = frame_count + 1;
            ScreenCapture.CaptureScreenshot(folderName + frame_count.ToString()+ ((int)Math.Round(transform.eulerAngles.y)).ToString() + ".png");

            if (frame_count == 200)
            {
                frame_count = 0;
                transform.RotateAround(target.transform.position, new Vector3(0, 1, 0), speed*Time.deltaTime);
            }
            
            counter = true;
        }
        else
        {
            target.transform.GetChild(i).gameObject.SetActive(false);
            transform.RotateAround(target.transform.position, new Vector3(0, 1, 0), speed*Time.deltaTime);
            if (counter == true && i < nChild)
            {
                i = i + 1;
                frame_count = 0;
                counter = false;
            }
            if (i == 9)
            {
                this.enabled = false;
            }
        }*/
    }
}
