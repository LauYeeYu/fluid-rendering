#define GL_SILENCE_DEPRECATION
#if __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include <unistd.h>
#include <iostream>
#include <vector>

#include "particle.h"

typedef Eigen::Matrix<unsigned int, 2, 1> Vector2ui;

// rendering projection parameters
const static unsigned int WINDOW_WIDTH = 500; // display window width
const static unsigned int WINDOW_HEIGHT = 500;

const static double VIEW_WIDTH = 50.0f; // actual view width
const static double VIEW_HEIGHT = WINDOW_HEIGHT * VIEW_WIDTH / WINDOW_WIDTH;
const static double VIEW_LAYER = VIEW_HEIGHT;

// global parameters
const static double visual_radius = 5;
const static unsigned int MAX_PARTICLES = 1200;
const static unsigned int fps = 50;
const static Eigen::Vector2d g(0.0f, -5.8f);
const static Eigen::Vector2d force(5.0, 1.0f);
static std::vector<Eigen::Vector3d> boundaries = std::vector<Eigen::Vector3d>();
const static double EPS = 1e-8f;
const static double EPS2 = EPS * EPS;

// solver parameters
const static unsigned int SOLVER_STEPS = 10;
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
static std::vector<Eigen::Vector2d> xlast(numParticles); // used to store last position for particles.
static std::vector<Eigen::Vector2d> xprojected(numParticles);

// griding parameters
static const double CELL_SIZE = H; // set to smoothing radius
static const unsigned int GRID_WIDTH = (unsigned int)(VIEW_WIDTH / CELL_SIZE);
static const unsigned int GRID_HEIGHT = (unsigned int)(VIEW_HEIGHT / CELL_SIZE);
static const unsigned int NUM_CELLS = GRID_WIDTH * GRID_HEIGHT;

// griding memory allocation
static std::vector<Particle*> grid(NUM_CELLS);
static std::vector<Eigen::Vector2i> gridIndices(MAX_PARTICLES);

const int MAX_Neighbor = 4000;
const int max_particle_in_grid = 500;


double lambdas[MAX_PARTICLES + 10];
Eigen::Vector2d delta_qs[MAX_PARTICLES + 10];
int num_particle_in_grid[GRID_WIDTH + 10][GRID_HEIGHT + 10];
int table_grid[GRID_WIDTH + 10][GRID_HEIGHT + 10][max_particle_in_grid];
int num_nb[MAX_PARTICLES + 10];
int table_nb[MAX_PARTICLES + 10][MAX_Neighbor];
int dx[] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
int dy[] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};


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

    Eigen::Vector2d start(0.25f * VIEW_WIDTH, 0.95f * VIEW_HEIGHT);
    double x0 = start(0);
    unsigned int num = sqrt(numParticles);
    double spacing = PARTICLE_RADIUS;
    std::cout << "initializing with " << num << " particles per row for " << num * num << " overall" << std::endl;
    for (int i = 0; i < numParticles; ++i) {
        auto pos_x = 10 + 0.75 * (i % 40);
        auto pos_y = 1 + 0.8 * (i / 40);
        particles.push_back(Particle(Eigen::Vector2d(pos_x, pos_y)));
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
        gridIndices[i] = Eigen::Vector2i(xind, yind);
    }
}

/// apply force
void ApplyExternalForces(void) 
{
    double d = 0;
    for (auto &p : particles)
        p.v += (g + d * force) * DT;
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

void PressureStep(void)
{
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        auto &pi = particles[i];

        Vector2ui ind = Vector2ui(gridIndices[i](0), gridIndices[i](1) * GRID_WIDTH);
        nh[i].numNeighbors = 0;

        double dens = 0.0f;
        double dens_proj = 0.0f;
        for (unsigned int ii = ind(0) - 1; ii <= ind(0) + 1; ii++)
            for (unsigned int jj = ind(1) - GRID_WIDTH; jj <= ind(1) + GRID_WIDTH; jj += GRID_WIDTH)
                for (Particle *pgrid = grid[ii + jj]; pgrid != NULL; pgrid = pgrid->n)
                {
                    const Particle &pj = *pgrid;
                    Eigen::Vector2d dx = pj.x - pi.x;
                    double r2 = dx.squaredNorm();
                    if (r2 < EPS2 || r2 > H * H)
                        continue;
                    double r = sqrt(r2);
                    double a = 1. - r / H;
                    dens += pj.m * a * a * a * KERN;
                    dens_proj += pj.m * a * a * a * a * KERN_NORM;
                    if (nh[i].numNeighbors < Neighborhood::MAX_NEIGHBORS)
                    {
                        nh[i].particles[nh[i].numNeighbors] = &pj;
                        nh[i].r[nh[i].numNeighbors] = r;
                        ++nh[i].numNeighbors;
                    }
                }

        pi.d = dens;
        pi.dv = dens_proj;
        pi.p = STIFFNESS * (dens - pi.m * REST_DENSITY);
        pi.pv = STIFF_APPROX * dens_proj;
    }
}

void Project(void)
{
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        auto &pi = particles[i];

        Eigen::Vector2d xx = pi.x;
        for (unsigned int j = 0; j < nh[i].numNeighbors; j++)
        {
            const Particle &pj = *nh[i].particles[j];
            double r = nh[i].r[j];
            Eigen::Vector2d dx = pj.x - pi.x;

            double a = 1. - r / H;
            double d = DT2 * ((pi.pv + pj.pv) * a * a * a * KERN_NORM + (pi.p + pj.p) * a * a * KERN) / 2.;

            // relaxation
            xx -= d * dx / (r * pi.m);

            // surface tension applies if the particles are of the same material
            // this would allow for extensibility of multi-phase
            if (pi.m == pj.m)
                xx += (SURFACE_TENSION / pi.m) * pj.m * a * a * KERN * dx;

            // linear and quadratic visc
            Eigen::Vector2d dv = pi.v - pj.v;
            double u = dv.dot(dx);
            if (u > 0)
            {
                u /= r;
                double a = 1 - r / H;
                double I = 0.5f * DT * a * (LINEAR_VISC * u + QUAD_VISC * u * u);
                xx -= I * dx * DT;
            }
        }
        xprojected[i] = xx;
    }
}

void Correct(void)
{
    for (unsigned int i = 0; i < particles.size(); i++)
    {
        auto &p = particles[i];
        p.x = xprojected[i];
        p.v = (p.x - xlast[i]) / DT;
    }
}

void EnforceBoundary(void)
{
    for (auto &p : particles)
        for (auto b : boundaries)
        {
            double d = p.x(0) * b(0) + p.x(1) * b(1) - b(2);
            if ((d = std::max(0., d)) < PARTICLE_RADIUS)
                p.v += (PARTICLE_RADIUS - d) * b.segment<2>(0) / DT;
        }
}

void InitGL(void)
{
    glClearColor(0.9f, 0.9f, 0.9f, 1);
    glEnable(GL_POINT_SMOOTH);
    glPointSize(visual_radius);
    glMatrixMode(GL_PROJECTION);
}

void Render(void)
{
    glClear(GL_COLOR_BUFFER_BIT);

    glLoadIdentity();
    glOrtho(0, VIEW_WIDTH, 0, VIEW_HEIGHT, 0, 1);

    glColor4f(0.2f, 0.6f, 1.0f, 1);
    glBegin(GL_POINTS);
    for (auto &p : particles)
        glVertex2f(p.x(0), p.x(1));
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

inline double poly6(Eigen::Vector2d x) {
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

inline Eigen::Vector2d spiky(Eigen::Vector2d x) {
    double r = x.norm();
    if (0 < r && r < H) 
        return spiky_Coe * pow(H - r, 2) / (h_6 * r) * x;
    return Eigen::Vector2d(0.0, 0.0);
}

inline double S_Corr(Eigen::Vector2d x) {
    auto upper = poly6(x);
    auto lower = poly6_scalar(S_Corr_delta_q * S_Corr_delta_q);
    auto m = upper / lower;
    return -S_Corr_k * m * m * m * m;
}

void boundary_check(Eigen::Vector2d &x) {
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
}

Eigen::Vector2i get_grid(Eigen::Vector2d x) {
    boundary_check(x);
    int xind = x(0) / CELL_SIZE;
    int yind = x(1) / CELL_SIZE;
    return Eigen::Vector2i(xind, yind);
}

void PBFNeighborSearch(void)
{
    for (int i = 0; i < GRID_WIDTH; ++i)
        for (int j = 0; j < GRID_HEIGHT; ++j) {
            num_particle_in_grid[i][j] = 0;
            for (int k = 0; k < max_particle_in_grid; ++k) {
                table_grid[i][j][k] = -1;
            }
        }
    for (int i = 0; i < (int) particles.size(); ++i) {
        num_nb[i] = 0;
        for (int j = 0; j < MAX_Neighbor; ++j) {
            table_nb[i][j] = -1;
        }
    }

    for (int id = 0; id < (int) particles.size(); ++id) {
        auto p = particles[id];
        auto p_grid = get_grid(p.x);
        if (p_grid(0) < 0 || p_grid(0) >= GRID_WIDTH || p_grid(1) < 0 || p_grid(1) >= GRID_HEIGHT) {
            printf("grid error\n");
            continue;
        }
        
        auto g_index = num_particle_in_grid[p_grid(0)][p_grid(1)]++;
        if (g_index >= max_particle_in_grid) {
            printf("grid overflow\n");
        }
        table_grid[p_grid(0)][p_grid(1)][g_index] = id;
        // if (id < 500) 
        //     std::cout << p_grid(0) << ' ' << p_grid(1) << ' ' << g_index << std::endl;
    }

    for (int id = 0; id < (int) particles.size(); ++id) {
        auto p = particles[id];
        auto p_grid = get_grid(p.x);
        for (int k = 0; k < 9; ++k) {
            int x = p_grid(0) + dx[k];
            int y = p_grid(1) + dy[k];
            if (x < 0 || x >= GRID_WIDTH || y < 0 || y >= GRID_HEIGHT) 
                continue;
            for (int i = 0; i < num_particle_in_grid[x][y]; ++i) {
                auto new_nb = table_grid[x][y][i];
                
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
        Eigen::Vector2d spiky_i = Eigen::Vector2d(0.0, 0.0);
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
        auto delta_q = Eigen::Vector2d(0.0, 0.0);
        auto pos = particles[id].x;
        for (int i = 0; i < num_nb[id]; ++i) {
            auto nb_index = table_nb[id][i];
            auto nb_pos = particles[nb_index].x;
            
            // std::cout << " FFF " << i << ' ' << pos << ' ' << nb_pos << std::endl;
            auto scorr = S_Corr(pos - nb_pos);
            auto left = lambdas[id] + lambdas[nb_index] + scorr;
            auto right = spiky(pos - nb_pos);
            delta_q += left * right / REST_DENSITY;
            // std::cout << "id: " << id << ' ' << lambdas[id] << ' ' << lambdas[nb_index] << ' ' << left << '#' << right << std::endl;
        }
        // std::cout << delta_q << std::endl;
        // exit(0);
        delta_qs[id] = delta_q;
    }
    // exit(0);
    for (int id = 0; id < (int) particles.size(); ++id) 
        particles[id].x += delta_qs[id];
    // cout << particles[0].x << endl;
    // exit(0);
}

void PBFupdate(void) {
    for (int id = 0; id < (int) particles.size(); ++id) {
        boundary_check(particles[id].x);
    }
    for (int id = 0; id < (int) particles.size(); ++id) {
        particles[id].v = (particles[id].x - xlast[id]) / DT;
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
    glutCreateWindow("PBF");
    glutDisplayFunc(Render);
    glutIdleFunc(Update);

    InitGL();
    InitParticles();

    glutMainLoop();
    return 0;
}
