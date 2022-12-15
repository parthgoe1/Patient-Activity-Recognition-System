using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class movement : MonoBehaviour
{
    public float horizontalSpeed = 4.0F;
    public float verticalSpeed = 4.0F;//Controls velocity multiplier
    public Animator animator;
    public float m = 3f,val;
    public bool pain;
    void Start()
    {
        pain = false;
        val = 0f;
    }

    // Update is called once per frame
 
         
    void Update()
    {
        float h = horizontalSpeed * Input.GetAxis("Mouse X");
        float v = verticalSpeed * Input.GetAxis("Mouse Y");
       
        animator.SetFloat("walking", Input.GetAxisRaw("Vertical"));
        transform.Rotate(0, h, 0);
        if (Input.GetKey(KeyCode.A))
        {
            transform.Translate(Vector3.left * m * Time.deltaTime);
        }
        if (Input.GetKey(KeyCode.D))
        {
            transform.Translate(Vector3.right * m * Time.deltaTime);
        }
        if (Input.GetKey(KeyCode.W))
        {
            transform.Translate(Vector3.forward * m * Time.deltaTime);
            
        }
       
        if (Input.GetKey(KeyCode.S))
        {
          //  transform.Translate(Vector3.back * m * Time.deltaTime);
            animator.SetBool("sitTrue", true);
        }
        else
        {
            animator.SetBool("sitTrue", false);
        }
        if (Input.GetKey(KeyCode.U))
        {
            animator.SetBool("upTrue", true);
        }
        else
        {
            animator.SetBool("upTrue", false);
        }

        if (Input.GetKey(KeyCode.P))
        {
           // if(!pain)
            {
                val = 1f;
                animator.SetFloat("PainTrue", val);
                //    pain = true;
            }
           
          //  if(pain)
            {
              //  val = 0f;
            //    pain = false;
            }
        }
        else
        {
            val = 0f;
            animator.SetFloat("PainTrue", val);
        }
        Debug.Log(val);
        










    }
}
