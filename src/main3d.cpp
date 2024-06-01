#define GL_SILENCE_DEPRECATION
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <unistd.h>
#include <iostream>
#include <vector>

#include "particle3d.h"

typedef Eigen::Matrix<unsigned int, 2, 1> Vector3ui;

// rendering projection parameters
const static unsigned int WINDOW_WIDTH = 500; // display window width
const static unsigned int WINDOW_HEIGHT = 500;

const static double VIEW_WIDTH = 20.0f; // actual view width
const static double VIEW_HEIGHT = WINDOW_HEIGHT * VIEW_WIDTH / WINDOW_WIDTH;
const static double VIEW_LAYER = VIEW_HEIGHT;

// global parameters
const static double visual_radius = 3;
const static unsigned int MAX_PARTICLES = 12000;
const static unsigned int fps = 30;
const static Eigen::Vector3d g(0.0f, 0.0f, -9.8f);
static std::vector<Eigen::Vector3d> boundaries = std::vector<Eigen::Vector3d>();
const static double EPS = 1e-2f;
const static double EPS2 = EPS * EPS;

// solver parameters
const static unsigned int SOLVER_STEPS = 5;
const static double REST_DENSITY = 1.0f; // 82.0f;
const static double STIFFNESS = 0.08f;
const static double STIFF_APPROX = 0.1f;
const static double SURFACE_TENSION = 0.0001f;
const static double LINEAR_VISC = 0.25f;
const static double QUAD_VISC = 0.5f;
const static double PARTICLE_RADIUS = 0.4f;
const static double H = 1.0f; // smoothing radius
const static double DT = 1.0f / fps;
const static double DT2 = DT * DT;
const static double KERN = 20. / (2. * M_PI * H * H);
const static double KERN_NORM = 30. / (2. * M_PI * H * H);

// global memory allocation
static unsigned int numParticles = MAX_PARTICLES;
static std::vector<Particle> particles = std::vector<Particle>();
static std::vector<Neighborhood> nh(numParticles);
static std::vector<Eigen::Vector3d> xlast(numParticles); // used to store last position for particles.
static std::vector<Eigen::Vector3d> xprojected(numParticles);

// griding parameters
static const double CELL_SIZE = H; // set to smoothing radius
static const unsigned int GRID_WIDTH = (unsigned int)(VIEW_WIDTH / CELL_SIZE);
static const unsigned int GRID_HEIGHT = (unsigned int)(VIEW_HEIGHT / CELL_SIZE);
static const unsigned int GRID_LAYER = (unsigned int)(VIEW_LAYER / CELL_SIZE);
static const unsigned int NUM_CELLS = GRID_WIDTH * GRID_HEIGHT;

// griding memory allocation
static std::vector<Particle*> grid(NUM_CELLS);
static std::vector<Eigen::Vector3i> gridIndices(MAX_PARTICLES);

const int MAX_Neighbor = 4000;
const int max_particle_in_grid = 4000;


double lambdas[MAX_PARTICLES + 10];
Eigen::Vector3d delta_qs[MAX_PARTICLES + 10], vorticity[MAX_PARTICLES + 10];
int num_particle_in_grid[GRID_WIDTH + 10][GRID_HEIGHT + 10][GRID_LAYER + 10];
int table_grid[GRID_WIDTH + 10][GRID_HEIGHT + 10][GRID_LAYER + 10][max_particle_in_grid];
int num_nb[MAX_PARTICLES + 10];
int table_nb[MAX_PARTICLES + 10][MAX_Neighbor];

int dx[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1};
int dy[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1};
int dz[] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1};

auto h_2 = H * H;
auto h_6 = h_2 * h_2 * h_2;
auto h_9 = h_6 * h_2 * H;

const double b_epsilon = 0.01;
const double poly6_Coe = 315.0 / (64 * 3.14);
const double spiky_Coe = -45.0 / 3.14;

const double lambda_epsilon = 100.0;

const double S_Corr_delta_q = 0.03;
const double S_Corr_k = 0.0001;

const double xsph_c = 0.01;
const double vorti_epsilon = 0.01;
const double g_del = 0.01;

using namespace std; // TODO: comment it

void GridInsert(void);

/// create particles
void InitParticles(void) 
{
    boundaries.push_back(Eigen::Vector3d(1, 0, 0));
    boundaries.push_back(Eigen::Vector3d(0, 1, 0));
    boundaries.push_back(Eigen::Vector3d(-1, 0, -VIEW_WIDTH));
    boundaries.push_back(Eigen::Vector3d(0, -1, -VIEW_HEIGHT));

    std::cout << "grid width: " << GRID_WIDTH << std::endl;
    std::cout << "grid height: " << GRID_HEIGHT << std::endl;
    std::cout << "cell size: " << CELL_SIZE << std::endl;
    std::cout << "num cells: " << NUM_CELLS << std::endl;

    unsigned int num = sqrt(numParticles);
    double spacing = PARTICLE_RADIUS;
    std::cout << "initializing with " << num << " particles per row for " << num * num << " overall" << std::endl;
    for (int i = 0; i < numParticles; ++i) {
        
        auto pos_x = 2 + 0.8 * (i % 20);
        auto pos_y = 2 + 0.8 * ((i % 400) / 20);
        auto pos_z = 1 + 0.8 * (i / 400);
        vorticity[i] = Eigen::Vector3d(0.0, 0.0, 0.0);
        particles.push_back(Particle(Eigen::Vector3d(pos_x, pos_y, pos_z)));
    }
    std::cout << "inserted " << particles.size() << " particles" << std::endl;

    GridInsert();
}

void GridInsert(void)
{
    for (auto &elem : grid)
        elem = NULL;
    for (auto &p : particles)
    {
        auto i = &p - &particles[0];
        unsigned int xind = p.x(0) / CELL_SIZE;
        unsigned int yind = p.x(1) / CELL_SIZE;
        xind = std::max(1U, std::min(GRID_WIDTH - 2, xind));
        yind = std::max(1U, std::min(GRID_HEIGHT - 2, yind));
        p.n = grid[xind + yind * GRID_WIDTH];
        grid[xind + yind * GRID_WIDTH] = &p;
        gridIndices[i] = Eigen::Vector3i(xind, yind);
    }
}

/// apply force
void ApplyExternalForces(void) 
{
    double d = 0;
    for (auto &p : particles)
        p.v += g * DT;
}

/// predict position TODO: boundary check for particles
void Integrate(void) 
{
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        auto &p = particles[i];
        xlast[i] = p.x;
        p.x += DT * p.v;
    }
}

void InitGL(void)
{
    glClearColor(0.9f, 0.9f, 0.9f, 1);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_DEPTH_TEST);
    glPointSize(visual_radius);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)WINDOW_WIDTH / (GLfloat)WINDOW_HEIGHT, 1.0, 100.0);
}

void Render(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    
    gluLookAt(30.0, 10.0, 15.0,   // eye position (x,y,z)
              0.0, 10.0, 5.0,      // look at pos (x,y,z)
              0.0, 0.0, 1.0);     // up-axis (x,y,z)

    glColor4f(0.2f, 0.6f, 1.0f, 1);
    // glEnable(GL_DEPTH_TEST); 
    // glLoadIdentity();
    // glOrtho(0, VIEW_WIDTH, 0, VIEW_HEIGHT, 0, 1);

    glColor4f(0.2f, 0.6f, 1.0f, 1);
    glBegin(GL_POINTS);
    // cout << particles[0].x << endl;
    for (auto &p : particles)
        glVertex3f(p.x(0), p.x(1), p.x(2));
        // glVertex2f(p.x(0), p.x(2));
    glEnd();
    glutSwapBuffers();
}

void PrintPositions(void)
{
    for (const auto &p : particles)
        std::cout << p.x << std::endl;
}

inline double rnd() {
    return (double)rand() / RAND_MAX;
}

inline double poly6(Eigen::Vector3d x) {
    double r2 = x.squaredNorm();
    if (0 < r2 && r2 < h_2) 
        return poly6_Coe * pow(h_2 - r2, 3) / h_9;
    return 0.0;
}

inline double poly6_scalar(double r2) {
    if (0 < r2 && r2 < h_2) 
        return poly6_Coe * pow(h_2 - r2, 3) / h_9;
    return 0.0;
}

inline Eigen::Vector3d spiky(Eigen::Vector3d x) {
    double r = x.norm();
    if (0 < r && r < H) 
        return spiky_Coe * pow(H - r, 2) / (h_6 * r) * x;
    return Eigen::Vector3d(0.0, 0.0, 0.0);
}

inline double S_Corr(Eigen::Vector3d x) {
    auto upper = poly6(x);
    auto lower = poly6_scalar(S_Corr_delta_q * S_Corr_delta_q);
    auto m = upper / lower;
    return -S_Corr_k * m * m * m * m;
}

void boundary_check(Eigen::Vector3d &x) {
    auto lower = PARTICLE_RADIUS;
    auto upper = VIEW_WIDTH - PARTICLE_RADIUS;
    
    if (x(0) < lower) 
        x(0) = lower + rnd() * EPS;
    else if (x(0) > upper)
        x(0) = upper - rnd() * EPS;
    
    upper = VIEW_HEIGHT - PARTICLE_RADIUS;
    if (x(1) < lower) 
        x(1) = lower + rnd() * EPS;
    else if (x(1) > upper)
        x(1) = upper - rnd() * EPS;

    upper = VIEW_LAYER - PARTICLE_RADIUS;
    if (x(2) < lower) 
        x(2) = lower + rnd() * EPS;
    else if (x(2) > upper)
        x(2) = upper - rnd() * EPS;
}

Eigen::Vector3i get_grid(Eigen::Vector3d x) {
    boundary_check(x);
    int xind = x(0) / CELL_SIZE;
    int yind = x(1) / CELL_SIZE;
    int zind = x(2) / CELL_SIZE;
    return Eigen::Vector3i(xind, yind, zind);
}

void PBFNeighborSearch(void)
{
    for (int i = 0; i < GRID_WIDTH; ++i)
        for (int j = 0; j < GRID_HEIGHT; ++j) 
          for (int k = 0; k < GRID_LAYER; ++k) {
            num_particle_in_grid[i][j][k] = 0;
            for (int l = 0; l < max_particle_in_grid; ++l) {
                table_grid[i][j][k][l] = -1;
            }
        }
    for (int i = 0; i < (int) particles.size(); ++i) {
        num_nb[i] = 0;
        vorticity[i] = Eigen::Vector3d(0.0, 0.0, 0.0);
        for (int j = 0; j < MAX_Neighbor; ++j) {
            table_nb[i][j] = -1;
        }
    }

    for (int id = 0; id < (int) particles.size(); ++id) {
        auto p = particles[id];
        auto p_grid = get_grid(p.x);
        if (p_grid(0) < 0 || p_grid(0) >= GRID_WIDTH ||
            p_grid(1) < 0 || p_grid(1) >= GRID_HEIGHT|| 
            p_grid(2) < 0 || p_grid(2) >= GRID_LAYER) {
            printf("grid error\n");
            continue;
        }
        
        auto g_index = num_particle_in_grid[p_grid(0)][p_grid(1)][p_grid(2)]++;
        if (g_index >= max_particle_in_grid) {
            printf("grid overflow\n");
        }
        table_grid[p_grid(0)][p_grid(1)][p_grid(2)][g_index] = id;
        // if (id < 500) 
        //     std::cout << p_grid(0) << ' ' << p_grid(1) << ' ' << g_index << std::endl;
    }

    for (int id = 0; id < (int) particles.size(); ++id) {
        auto p = particles[id];
        auto p_grid = get_grid(p.x);
        for (int k = 0; k < 27; ++k) {
            int x = p_grid(0) + dx[k];
            int y = p_grid(1) + dy[k];
            int z = p_grid(2) + dz[k];
            if (x < 0 || x >= GRID_WIDTH || y < 0 || y >= GRID_HEIGHT || z < 0 || z >= GRID_LAYER) 
                continue;
            for (int i = 0; i < num_particle_in_grid[x][y][z]; ++i) {
                auto new_nb = table_grid[x][y][z][i];
                
                auto n_index = (num_nb[id] ++); // calculate the number of neighbors
                if (n_index >= MAX_Neighbor) {
                    printf("neighbor overflow\n");
                }
                table_nb[id][n_index] = new_nb;
            }
        }
    }
}


void PBFsolve(void) {
    for (int id = 0; id < (int) particles.size(); ++id) {
        auto pos = particles[id].x;
        double lower_sum = 0.0;
        double p_i = 0.0;
        Eigen::Vector3d spiky_i = Eigen::Vector3d(0.0, 0.0, 0.0);
        for (int i = 0; i < num_nb[id]; ++i) {
            auto nb_index = table_nb[id][i];
            auto nb_pos = particles[nb_index].x;
            p_i += Particle::PARTICLE_MASS * poly6(pos - nb_pos);

            auto s = spiky(pos - nb_pos) / REST_DENSITY;
            spiky_i += s;
            lower_sum += s.squaredNorm();
        }
        auto constraint = (p_i / REST_DENSITY) - 1.0;
        lower_sum += spiky_i.squaredNorm();
        lambdas[id] = -constraint / (lower_sum + lambda_epsilon);
    }

    for (int id = 0; id < (int) particles.size(); ++id) {
        auto delta_q = Eigen::Vector3d(0.0, 0.0, 0.0);
        auto pos = particles[id].x;
        for (int i = 0; i < num_nb[id]; ++i) {
            auto nb_index = table_nb[id][i];
            auto nb_pos = particles[nb_index].x;
            
            auto scorr = S_Corr(pos - nb_pos);
            auto left = lambdas[id] + lambdas[nb_index] + scorr;
            auto right = spiky(pos - nb_pos);
            delta_q += left * right / REST_DENSITY;
        }
        delta_qs[id] = delta_q;
    }
    for (int id = 0; id < (int) particles.size(); ++id) 
        particles[id].x += delta_qs[id];
}

void PBFupdate(void) {
    for (int id = 0; id < (int) particles.size(); ++id) {
        boundary_check(particles[id].x);
    }
    for (int id = 0; id < (int) particles.size(); ++id) {
        particles[id].v = (particles[id].x - xlast[id]) / DT;
    }
    
    for (int id = 0; id < (int) particles.size(); ++id) {
        auto pos = particles[id].x;
        auto xsph_sum = Eigen::Vector3d(0.0, 0.0, 0.0);
        auto omega_sum = Eigen::Vector3d(0.0, 0.0, 0.0);
        auto dx_sum = Eigen::Vector3d(0.0, 0.0, 0.0);
        auto dy_sum = Eigen::Vector3d(0.0, 0.0, 0.0);
        auto dz_sum = Eigen::Vector3d(0.0, 0.0, 0.0);

        auto n_dx_sum = Eigen::Vector3d(0.0, 0.0, 0.0);
        auto n_dy_sum = Eigen::Vector3d(0.0, 0.0, 0.0);
        auto n_dz_sum = Eigen::Vector3d(0.0, 0.0, 0.0);

        auto dx = Eigen::Vector3d(g_del, 0.0, 0.0);
        auto dy = Eigen::Vector3d(0.0, g_del, 0.0);
        auto dz = Eigen::Vector3d(0.0, 0.0, g_del);
        for (int i = 0; i < num_nb[id]; ++i) {
            auto nb_index = table_nb[id][i];
            auto nb_pos = particles[nb_index].x;
            auto v_ij = particles[nb_index].v - particles[id].v;
            auto dist = pos - nb_pos;

            omega_sum += v_ij.cross(spiky(dist));
            dx_sum += v_ij.cross(spiky(dist + dx));
            dy_sum += v_ij.cross(spiky(dist + dy));
            dz_sum += v_ij.cross(spiky(dist + dz));
            n_dx_sum += v_ij.cross(spiky(dist - dx));
            n_dy_sum += v_ij.cross(spiky(dist - dy));
            n_dz_sum += v_ij.cross(spiky(dist - dz));
            auto poly = poly6(dist);
            xsph_sum += poly * v_ij;
        }
        auto n_x = (dx_sum.norm() - n_dx_sum.norm()) / (2 * g_del);
        auto n_y = (dy_sum.norm() - n_dy_sum.norm()) / (2 * g_del);
        auto n_z = (dz_sum.norm() - n_dz_sum.norm()) / (2 * g_del);
        auto n = Eigen::Vector3d(n_x, n_y, n_z);
        auto big_n = n.normalized();
        if (! (omega_sum.norm() == 0)) {
            particles[id].v = vorti_epsilon * big_n.cross(omega_sum);
        }
        xsph_sum *= xsph_c;
        particles[id].v += xsph_sum;
    }
}

void Update(void) {
    ApplyExternalForces();
    Integrate();
    PBFNeighborSearch();
    for (int i = 0; i < SOLVER_STEPS; i++) {
        PBFsolve();
    }
    
    PBFupdate();
    glutPostRedisplay();
}

int main(int argc, char **argv)
{
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutCreateWindow("PBF");
    glutDisplayFunc(Render);
    glutIdleFunc(Update);

    InitGL();
    InitParticles();

    glutMainLoop();
    return 0;
}
