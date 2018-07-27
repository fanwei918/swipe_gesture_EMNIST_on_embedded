
int x[52] = {
    216,215,215,211,210,204,175,163,115,107,97,64,59,52,40,39,37,40,41,43,61,68,86,99,103,110,112,120,133,136,138,147,148,151,152,153,153,153,153,154,154,154,154,153,151,151,151,151,151,151,151,151};
int y[52] = {
    229,229,229,230,230,230,229,228,225,225,224,219,218,217,217,217,216,212,211,209,194,189,170,150,147,138,134,121,107,102,100,91,89,85,85,84,83,102,113,143,153,188,200,258,297,307,361,329,350,358,360,360};


int n_points = sizeof(x)/sizeof(x[0]);

int mnist_vec[28*28];

void mnist_from_trace(int* x, int* y, int n_points, int* mnist_vec)
{
    
    int n;
    
    int xmax = 0;
    int xmin = 500;
    int ymax = 0;
    int ymin = 500;
    for(n=0;n<n_points;n++)
    {
        xmax = (x[n]>xmax?x[n]:xmax);
        xmin = (x[n]<xmin?x[n]:xmin);
        ymax = (y[n]>ymax?y[n]:ymax);
        ymin = (y[n]<ymin?y[n]:ymin);
        
    }
    
    int dx = xmax - xmin;
    int dy = ymax - ymin;
    int ds = dy>dx?dy:dx;
    
    int margin = 60;
    ds += margin;
    
    int n_mul;
    for(n_mul=1;n_mul<18;n_mul++)
    {
        if (ds < n_mul*28)
            break;
    }
    
    int x_start = (xmin+xmax)/2 - ds/2;
    int y_start = (ymin+ymax)/2 - ds/2;
    
    
    int x_new[n_points];
    int y_new[n_points];
    for(n=0;n<n_points;n++)
    {
        x_new[n] = x[n] - x_start;
        y_new[n] = y[n] - y_start;
    }
    
    
    for (n=0;n<n_points;n++)
    {
        int x_in_minst_mat = y_new[n]/n_mul;
        int y_in_minst_mat = x_new[n]/n_mul;
        
        mnist_vec[x_in_minst_mat*28+y_in_minst_mat] = 256;
    }
    
}
