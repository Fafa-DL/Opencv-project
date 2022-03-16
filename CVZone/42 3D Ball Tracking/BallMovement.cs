using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BallMovement : MonoBehaviour
{
    // Start is called before the first frame update

    public UDPReceive udpReceive;

    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {

        string data = udpReceive.data;
        data = data.Remove(0, 1);
        data = data.Remove(data.Length-1, 1);
        //(255,361,50012) example data (x,y,area)
        string[] info = data.Split(',');

        float x = 5- float.Parse(info[0])/100;
        float y = float.Parse(info[1])/100;
        float z = -10 + float.Parse(info[2])/1000;

        gameObject.transform.localPosition = new Vector3(x, y, z);


    }
}