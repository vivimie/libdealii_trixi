#include <fstream>

#include <deal.II/base/timer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_dgq.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/tools.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_transfer_global_coarsening.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/vector_tools.h>

#include "intercomm.h"


// Set debug mode for verbose printing of function calls
// #define DEBUG_VERBOSE 1


namespace PoissonModule
{
  using namespace dealii;



  template <int dim, typename number = double>
  class LaplaceOperator : public EnableObserverPointer
  {
  public:
    using value_type = number;
    using VectorType = LinearAlgebra::distributed::Vector<number>;

    LaplaceOperator(){};

    void
    initialize(const Mapping<dim> &             mapping,
               const DoFHandler<dim> &          dof_handler,
               const AffineConstraints<number> &constraints,
               const unsigned int               n_q_points_1d,
               const DoFHandler<dim> &          auxiliary_dof_handler,
               const unsigned int level = numbers::invalid_unsigned_int)
    {
      fe_degree = dof_handler.get_fe().degree;

      const QGauss<1> quad(n_q_points_1d);
      const QGauss<1> quad_aux(auxiliary_dof_handler.get_fe().degree + 1);
      typename MatrixFree<dim, number>::AdditionalData addit_data;
      addit_data.tasks_parallel_scheme =
        MatrixFree<dim, number>::AdditionalData::none;
      addit_data.mg_level = level;
      addit_data.mapping_update_flags_inner_faces =
        (update_gradients | update_JxW_values);
      addit_data.mapping_update_flags_boundary_faces =
        (update_gradients | update_JxW_values);
      AffineConstraints<number> dummy;

      data.reinit(mapping,
                  std::vector<const DoFHandler<dim> *>{
                    {&dof_handler, &auxiliary_dof_handler}},
                  std::vector<const AffineConstraints<number> *>{
                    {&constraints, &dummy}},
                  std::vector<Quadrature<1>>{{quad, quad_aux}},
                  addit_data);
    }

    void
    initialize(const Mapping<dim> &             mapping,
               const DoFHandler<dim> &          dof_handler,
               const AffineConstraints<number> &constraints,
               const unsigned int               n_q_points_1d,
               const unsigned int level = numbers::invalid_unsigned_int)
    {
      fe_degree = dof_handler.get_fe().degree;

      const QGauss<1>                                  quad(n_q_points_1d);
      typename MatrixFree<dim, number>::AdditionalData addit_data;
      addit_data.tasks_parallel_scheme =
        MatrixFree<dim, number>::AdditionalData::none;
      addit_data.mg_level = level;
      if (dof_handler.get_fe().dofs_per_vertex == 0)
        {
          addit_data.mapping_update_flags_inner_faces =
            (update_gradients | update_JxW_values);
          addit_data.mapping_update_flags_boundary_faces =
            (update_gradients | update_JxW_values);
        }

      data.reinit(mapping, dof_handler, constraints, quad, addit_data);
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      if (data.get_dof_handler().get_fe().dofs_per_vertex > 0)
        {
          // Continuous elements
          data.cell_loop(&LaplaceOperator::local_apply, this, dst, src, true);
          for (const auto i : data.get_constrained_dofs())
            dst.local_element(i) = src.local_element(i);
        }
      else
        {
          data.loop(&LaplaceOperator::local_apply,
                    &LaplaceOperator::local_apply_face,
                    &LaplaceOperator::local_apply_boundary,
                    this,
                    dst,
                    src,
                    true,
                    MatrixFree<dim, number>::DataAccessOnFaces::gradients,
                    MatrixFree<dim, number>::DataAccessOnFaces::gradients);
        }
    }

    void
    vmult(VectorType &      dst,
          const VectorType &src,
          const std::function<void(const unsigned int, const unsigned int)>
            &operation_before_loop,
          const std::function<void(const unsigned int, const unsigned int)>
            &operation_after_loop) const
    {
      // interleaving between vector operations and data only supported for
      // continuous FEM, not DG
      if (data.get_dof_handler().get_fe().dofs_per_vertex > 0)
        {
          // Continuous elements
          data.cell_loop(&LaplaceOperator::local_apply,
                         this,
                         dst,
                         src,
                         operation_before_loop,
                         operation_after_loop);
        }
      else
        {
          operation_before_loop(0, dst.locally_owned_size());
          vmult(dst, src);
          operation_after_loop(0, dst.locally_owned_size());
        }
    }

    void
    Tvmult(VectorType &dst, const VectorType &src) const
    {
      vmult(dst, src);
    }

    number
    el(const types::global_dof_index, const types::global_dof_index) const
    {
      AssertThrow(false, ExcNotImplemented());
      return number(0.);
    }

    types::global_dof_index
    m() const
    {
      return data.get_vector_partitioner()->size();
    }

    types::global_dof_index
    n() const
    {
      return data.get_vector_partitioner()->size();
    }

    void
    initialize_dof_vector(VectorType &       vector,
                          const unsigned int component = 0) const
    {
      data.initialize_dof_vector(vector, component);
    }

    void
    compute_inverse_diagonal()
    {
      inverse_diagonal_entries = std::make_shared<DiagonalMatrix<VectorType>>();
      data.initialize_dof_vector(inverse_diagonal_entries->get_vector());
      unsigned int dummy = 0;
      if (data.get_dof_handler().get_fe().dofs_per_vertex > 0)
        MatrixFreeTools::
          compute_diagonal<dim, -1, 0, 1, number, VectorizedArray<number>>(
            data, inverse_diagonal_entries->get_vector(), [](auto &eval) {
              eval.evaluate(EvaluationFlags::gradients);
              for (unsigned int q = 0; q < eval.n_q_points; ++q)
                eval.submit_gradient(eval.get_gradient(q), q);
              eval.integrate(EvaluationFlags::gradients);
            });
      else
        data.loop(&LaplaceOperator::local_diagonal_cell,
                  &LaplaceOperator::local_diagonal_face,
                  &LaplaceOperator::local_diagonal_boundary,
                  this,
                  inverse_diagonal_entries->get_vector(),
                  dummy);

      for (number &entry : inverse_diagonal_entries->get_vector())
        if (std::abs(entry) > 1e-10)
          entry = 1. / entry;
        else
          entry = 1.;
    }

    const std::shared_ptr<DiagonalMatrix<VectorType>> &
    get_matrix_diagonal_inverse() const
    {
      return inverse_diagonal_entries;
    }

    const MatrixFree<dim, number> &
    get_matrix_free() const
    {
      return data;
    }

  private:
    void
    local_apply(const MatrixFree<dim, number> &              data,
                VectorType &                                 dst,
                const VectorType &                           src,
                const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, 1, number> eval(data);

      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          eval.reinit(cell);
          eval.gather_evaluate(src, EvaluationFlags::gradients);
          for (unsigned int q = 0; q < eval.n_q_points; ++q)
            eval.submit_gradient(eval.get_gradient(q), q);
          eval.integrate_scatter(EvaluationFlags::gradients, dst);
        }
    }

    void
    local_apply_face(
      const MatrixFree<dim, number> &              data,
      VectorType &                                 dst,
      const VectorType &                           src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, 1, number> eval(data, true);
      FEFaceEvaluation<dim, -1, 0, 1, number> eval_neighbor(data, false);

      for (unsigned int face = face_range.first; face < face_range.second;
           ++face)
        {
          eval.reinit(face);
          eval_neighbor.reinit(face);

          eval.gather_evaluate(src,
                               EvaluationFlags::values |
                                 EvaluationFlags::gradients);
          eval_neighbor.gather_evaluate(src,
                                        EvaluationFlags::values |
                                          EvaluationFlags::gradients);
          VectorizedArray<number> sigmaF =
            (std::abs((eval.normal_vector(0) *
                       eval.inverse_jacobian(0))[dim - 1]) +
             std::abs((eval.normal_vector(0) *
                       eval_neighbor.inverse_jacobian(0))[dim - 1])) *
            (number)(std::max(fe_degree, 1) * (fe_degree + 1.0));

          for (unsigned int q = 0; q < eval.n_q_points; ++q)
            {
              VectorizedArray<number> average_value =
                (eval.get_value(q) - eval_neighbor.get_value(q)) * 0.5;
              VectorizedArray<number> average_valgrad =
                eval.get_normal_derivative(q) +
                eval_neighbor.get_normal_derivative(q);
              average_valgrad =
                average_value * 2. * sigmaF - average_valgrad * 0.5;
              eval.submit_normal_derivative(-average_value, q);
              eval_neighbor.submit_normal_derivative(-average_value, q);
              eval.submit_value(average_valgrad, q);
              eval_neighbor.submit_value(-average_valgrad, q);
            }
          eval.integrate_scatter(EvaluationFlags::values |
                                   EvaluationFlags::gradients,
                                 dst);
          eval_neighbor.integrate_scatter(EvaluationFlags::values |
                                            EvaluationFlags::gradients,
                                          dst);
        }
    }

    void
    local_apply_boundary(
      const MatrixFree<dim, number> &              data,
      VectorType &                                 dst,
      const VectorType &                           src,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      // TODO: Right now, we assume Dirichlet conditions everywhere
      FEFaceEvaluation<dim, -1, 0, 1, number> eval(data, true);
      for (unsigned int face = face_range.first; face < face_range.second;
           ++face)
        {
          eval.reinit(face);
          eval.gather_evaluate(src,
                               EvaluationFlags::values |
                                 EvaluationFlags::gradients);
          VectorizedArray<number> sigmaF =
            std::abs(
              (eval.normal_vector(0) * eval.inverse_jacobian(0))[dim - 1]) *
            number(std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

          for (unsigned int q = 0; q < eval.n_q_points; ++q)
            {
              VectorizedArray<number> average_value = eval.get_value(q);
              VectorizedArray<number> average_valgrad =
                -eval.get_normal_derivative(q);
              average_valgrad += average_value * sigmaF * 2.0;
              eval.submit_normal_derivative(-average_value, q);
              eval.submit_value(average_valgrad, q);
            }

          eval.integrate_scatter(EvaluationFlags::values |
                                   EvaluationFlags::gradients,
                                 dst);
        }
    }

    void
    local_diagonal_cell(
      const MatrixFree<dim, number> &data,
      VectorType &                   dst,
      const unsigned int &,
      const std::pair<unsigned int, unsigned int> &cell_range) const
    {
      FEEvaluation<dim, -1, 0, 1, number>    eval(data);
      AlignedVector<VectorizedArray<number>> local_diagonal_vector(
        eval.dofs_per_cell);

      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          eval.reinit(cell);

          // Compute diagonal by applying operator onto unit vectors
          for (unsigned int i = 0; i < eval.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < eval.dofs_per_cell; ++j)
                eval.begin_dof_values()[j] = VectorizedArray<number>();
              eval.begin_dof_values()[i] = 1.;
              eval.evaluate(EvaluationFlags::gradients);
              for (unsigned int q = 0; q < eval.n_q_points; ++q)
                eval.submit_gradient(eval.get_gradient(q), q);
              eval.integrate(EvaluationFlags::gradients);

              // from the full column of the matrix, only pick the i-th entry
              // (diagonal)
              local_diagonal_vector[i] = eval.begin_dof_values()[i];
            }
          for (unsigned int i = 0; i < eval.dofs_per_cell; ++i)
            eval.begin_dof_values()[i] = local_diagonal_vector[i];
          eval.distribute_local_to_global(dst);
        }
    }

    void
    local_diagonal_face(
      const MatrixFree<dim, number> &data,
      VectorType &                   dst,
      const unsigned int &,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, 1, number> eval(data, true);
      FEFaceEvaluation<dim, -1, 0, 1, number> eval_outer(data, false);
      AlignedVector<VectorizedArray<number>>  local_diagonal_vector(
        eval.dofs_per_cell);

      for (unsigned int face = face_range.first; face < face_range.second;
           ++face)
        {
          eval.reinit(face);
          eval_outer.reinit(face);

          VectorizedArray<number> sigmaF =
            (std::abs((eval.normal_vector(0) *
                       eval.inverse_jacobian(0))[dim - 1]) +
             std::abs((eval.normal_vector(0) *
                       eval_outer.inverse_jacobian(0))[dim - 1])) *
            (number)(2.0 * std::max(fe_degree, 1) * (fe_degree + 1.0));

          // Compute coupling eval - eval
          for (unsigned int j = 0; j < eval.dofs_per_cell; ++j)
            eval_outer.begin_dof_values()[j] = VectorizedArray<number>();
          eval_outer.evaluate(EvaluationFlags::values |
                              EvaluationFlags::gradients);
          for (unsigned int i = 0; i < eval.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < eval.dofs_per_cell; ++j)
                eval.begin_dof_values()[j] = VectorizedArray<number>();
              eval.begin_dof_values()[i] = 1.;
              eval.evaluate(EvaluationFlags::values |
                            EvaluationFlags::gradients);

              for (unsigned int q = 0; q < eval.n_q_points; ++q)
                {
                  VectorizedArray<number> average_value =
                    (eval.get_value(q) - eval_outer.get_value(q)) * 0.5;
                  VectorizedArray<number> average_valgrad =
                    eval.get_normal_derivative(q) +
                    eval_outer.get_normal_derivative(q);
                  average_valgrad =
                    average_value * 2. * sigmaF - average_valgrad * 0.5;
                  eval.submit_normal_derivative(-average_value, q);
                  eval.submit_value(average_valgrad, q);
                }
              eval.integrate(EvaluationFlags::values |
                             EvaluationFlags::gradients);
              local_diagonal_vector[i] = eval.begin_dof_values()[i];
            }
          for (unsigned int i = 0; i < eval.dofs_per_cell; ++i)
            eval.begin_dof_values()[i] = local_diagonal_vector[i];
          eval.distribute_local_to_global(dst);

          // Compute coupling eval_outer - eval_outer
          for (unsigned int j = 0; j < eval.dofs_per_cell; ++j)
            eval.begin_dof_values()[j] = VectorizedArray<number>();
          eval.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
          for (unsigned int i = 0; i < eval.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < eval.dofs_per_cell; ++j)
                eval_outer.begin_dof_values()[j] = VectorizedArray<number>();
              eval_outer.begin_dof_values()[i] = 1.;
              eval_outer.evaluate(EvaluationFlags::values |
                                  EvaluationFlags::gradients);

              for (unsigned int q = 0; q < eval.n_q_points; ++q)
                {
                  VectorizedArray<number> average_value =
                    (eval.get_value(q) - eval_outer.get_value(q)) * 0.5;
                  VectorizedArray<number> average_valgrad =
                    eval.get_normal_derivative(q) +
                    eval_outer.get_normal_derivative(q);
                  average_valgrad =
                    average_value * 2. * sigmaF - average_valgrad * 0.5;
                  eval_outer.submit_normal_derivative(-average_value, q);
                  eval_outer.submit_value(-average_valgrad, q);
                }
              eval_outer.integrate(EvaluationFlags::values |
                                   EvaluationFlags::gradients);
              local_diagonal_vector[i] = eval_outer.begin_dof_values()[i];
            }
          for (unsigned int i = 0; i < eval.dofs_per_cell; ++i)
            eval_outer.begin_dof_values()[i] = local_diagonal_vector[i];
          eval_outer.distribute_local_to_global(dst);
        }
    }

    void
    local_diagonal_boundary(
      const MatrixFree<dim, number> &data,
      VectorType &                   dst,
      const unsigned int &,
      const std::pair<unsigned int, unsigned int> &face_range) const
    {
      FEFaceEvaluation<dim, -1, 0, 1, number> eval(data);
      AlignedVector<VectorizedArray<number>>  local_diagonal_vector(
        eval.dofs_per_cell);

      for (unsigned int face = face_range.first; face < face_range.second;
           ++face)
        {
          eval.reinit(face);

          VectorizedArray<number> sigmaF =
            std::abs(
              (eval.normal_vector(0) * eval.inverse_jacobian(0))[dim - 1]) *
            number(2.0 * std::max(fe_degree, 1) * (fe_degree + 1.0)) * 2.0;

          for (unsigned int i = 0; i < eval.dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < eval.dofs_per_cell; ++j)
                eval.begin_dof_values()[j] = VectorizedArray<number>();
              eval.begin_dof_values()[i] = 1.;
              eval.evaluate(EvaluationFlags::values |
                            EvaluationFlags::gradients);

              for (unsigned int q = 0; q < eval.n_q_points; ++q)
                {
                  VectorizedArray<number> average_value = eval.get_value(q);
                  VectorizedArray<number> average_valgrad =
                    -eval.get_normal_derivative(q);
                  average_valgrad += average_value * sigmaF * 2.0;
                  eval.submit_normal_derivative(-average_value, q);
                  eval.submit_value(average_valgrad, q);
                }

              eval.integrate(EvaluationFlags::values |
                             EvaluationFlags::gradients);
              local_diagonal_vector[i] = eval.begin_dof_values()[i];
            }
          for (unsigned int i = 0; i < eval.dofs_per_cell; ++i)
            eval.begin_dof_values()[i] = local_diagonal_vector[i];
          eval.distribute_local_to_global(dst);
        }
    }

    MatrixFree<dim, number>                     data;
    std::shared_ptr<DiagonalMatrix<VectorType>> inverse_diagonal_entries;
    int                                         fe_degree;
  };



  template <typename Number>
  void
  make_zero_mean(const std::vector<unsigned int> &           constrained_dofs,
                 LinearAlgebra::distributed::Vector<Number> &vec)
  {
    // first set constrained entries to zero and count the number of the
    // constrained dofs
    unsigned int local_size = vec.locally_owned_size();
    for (const unsigned int index : constrained_dofs)
      {
        vec.local_element(index) = 0.;
        --local_size;
      }

    // Now rescale the mean value computed among all vector entries to the
    // vector size without constraints, which gets then subtracted from the
    // vector as a constant vector
    double mean_value = vec.mean_value();
    vec.add(-mean_value * vec.size() /
            Utilities::MPI::sum(local_size, vec.get_mpi_communicator()));

    // Finally, set constraints to zero again and ensure that the vector has
    // zero mean
    for (const unsigned int index : constrained_dofs)
      vec.local_element(index) = 0.;

    Assert(std::abs(vec.mean_value()) <
             std::numeric_limits<Number>::epsilon() * vec.size(),
           ExcInternalError());
  }



  // We want to define our own class for the coarse solver rather than using
  // one of deal.II's classes (MGCoarseGridApplySmoother is similar) because
  // we want to control the mean value of the coarse solution - in multigrid,
  // we only need to adjust the rhs and solution on the coarse level for
  // zero-mean because it contains this mode already, and thus propagates to
  // finer levels for Krylov methods that preserve invariant subspaces
  template <class VectorType = LinearAlgebra::distributed::Vector<double>>
  class MGCoarseSolverPeriodicity : public MGCoarseGridBase<VectorType>
  {
  public:
    MGCoarseSolverPeriodicity() = default;

    void
    clear()
    {
      coarse_smooth = nullptr;
    }

    void
    initialize(const MGSmootherBase<VectorType> &coarse_smooth,
               const std::vector<unsigned int> & constrained_dofs,
               const bool                        apply_periodicity)
    {
      this->coarse_smooth     = &coarse_smooth;
      this->constrained_dofs  = &constrained_dofs;
      this->apply_periodicity = apply_periodicity;
    }

    void
    operator()(const unsigned int level,
               VectorType &       dst,
               const VectorType & src) const override
    {
      if (apply_periodicity)
        {
          src_copy.reinit(src, true);
          src_copy.copy_locally_owned_data_from(src);
          make_zero_mean(*constrained_dofs, src_copy);
          coarse_smooth->apply(level, dst, src_copy);
          make_zero_mean(*constrained_dofs, dst);
        }
      else
        coarse_smooth->apply(level, dst, src);
    }

  private:
    ObserverPointer<const MGSmootherBase<VectorType>> coarse_smooth;
    const std::vector<unsigned int> *              constrained_dofs;

    mutable VectorType src_copy;
    bool               apply_periodicity;
  };



  template <int dim>
  class InitNaN : public Function<dim>
  {
  public:
    InitNaN()
      : Function<dim>(1)
    {}

    virtual double
    value(const Point<dim> & /* p */, const unsigned int) const override
    {
      return std::numeric_limits<double>::signaling_NaN();
    }
  };



  class WrapperBase
  {
  public:
    using VectorTyped = LinearAlgebra::distributed::Vector<double>;
    using VectorTypef = LinearAlgebra::distributed::Vector<float>;

    virtual ~WrapperBase() = default;

    virtual void solve() = 0;
    virtual void output_grid() = 0;
    virtual void adapt_grid() = 0;
    virtual void repartition_grid() = 0;
    virtual unsigned int get_num_local_active_cells() const = 0;
    virtual unsigned int get_mesh_checksum() const = 0;
    virtual std::array<double *, 3> compute_grad_u() = 0;
    virtual void print_timings() const = 0;

    unsigned int maxiters;
    double abstol;
    IndexSet locally_owned_dofs;

    std::array<double *, 3> pointers_grad_u;

    VectorTyped vector_f; // source term of the Poisson problem (low degree via Trixi.jl)
    VectorTyped vector_u; // solution computed for the Poisson problem (low degree via Trixi.jl)
    Vector<int> vector_amr_indicator; // AMR indicator value
  };

  template <int dim>
  class Wrapper : public WrapperBase
  {
    using TriaType = parallel::distributed::Triangulation<dim>;

  public:
    Wrapper(int polydeg, const MeshSettings3D meshsettings, int maxiters, double abstol)
      : triangulation(MPI_COMM_WORLD, TriaType::MeshSmoothing::none, meshsettings.repartition_after_adapt ? TriaType::Settings::default_setting : TriaType::Settings::no_automatic_repartitioning)
      , mapping(meshsettings.polydeg)
      , fe(polydeg)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer_output(pcout, TimerOutput::never, TimerOutput::wall_times)
    {
      this->maxiters = maxiters;
      this->abstol = abstol;
      Point<dim> p1(meshsettings.coordinates_min[0],
                    meshsettings.coordinates_min[1],
                    meshsettings.coordinates_min[2]);
      Point<dim> p2(meshsettings.coordinates_max[0],
                    meshsettings.coordinates_max[1],
                    meshsettings.coordinates_max[2]);
      std::vector<unsigned int> reps{meshsettings.trees_per_dimension[0],
                                     meshsettings.trees_per_dimension[1],
                                     meshsettings.trees_per_dimension[2]};
      create_mesh_and_unknowns(reps, p1, p2, meshsettings.initial_refinement_level);
    }

    Wrapper(int polydeg, const MeshSettings2D meshsettings, int maxiters, double abstol)
      : triangulation(MPI_COMM_WORLD, TriaType::MeshSmoothing::none, meshsettings.repartition_after_adapt ? TriaType::Settings::default_setting : TriaType::Settings::no_automatic_repartitioning)
      , mapping(meshsettings.polydeg)
      , fe(polydeg)
      , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      , timer_output(pcout, TimerOutput::never, TimerOutput::wall_times)
    {
      this->maxiters = maxiters;
      this->abstol = abstol;
      Point<dim> p1(meshsettings.coordinates_min[0],
                    meshsettings.coordinates_min[1]);
      Point<dim> p2(meshsettings.coordinates_max[0],
                    meshsettings.coordinates_max[1]);
      std::vector<unsigned int> reps{meshsettings.trees_per_dimension[0],
                                     meshsettings.trees_per_dimension[1]};
      create_mesh_and_unknowns(reps, p1, p2, meshsettings.initial_refinement_level);
    }

    void create_mesh_and_unknowns(const std::vector<unsigned int> repetitions,
                                  const Point<dim> &p1,
                                  const Point<dim> &p2,
                                  const unsigned int initial_refinement_level)
    {
      {
        TimerOutput::Scope timer(timer_output, "Setup grid");
        GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions, p1, p2);

        // set boundary ids on boundaries to the number of the face
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
             ++face)
          triangulation.begin()->face(face)->set_all_boundary_ids(face);
        std::vector<GridTools::PeriodicFacePair<
          typename Triangulation<dim>::cell_iterator>>
          periodic_faces;
        for (unsigned int d = 0; d < dim; ++d)
          GridTools::collect_periodic_faces(
            triangulation, 2 * d, 2 * d + 1, d, periodic_faces);
        triangulation.add_periodicity(periodic_faces);

        triangulation.refine_global(initial_refinement_level);
        triangulation.repartition();

        if constexpr (gridoutput)
           output_grid();
      }

      setup_unknowns();
      setup_multigrid_operators_and_transfer();
    }

  private:
    // renumber dofs to follow Morton order (z-curve as p4est) for
    // compatibility with trixi
    void
    morton_reorder_dofs()
    {
      std::vector<typename DoFHandler<dim>::active_cell_iterator> cell_order;
      cell_order.reserve(triangulation.n_locally_owned_active_cells());
      for (const auto &coarse_cell : dof_handler.cell_iterators_on_level(0))
        {
          // recursively traverse into children
          process_cells_recursively_on_tree(coarse_cell, [&](auto &cell) {
            cell_order.push_back(cell);
          });
        }
      DoFRenumbering::cell_wise(dof_handler, cell_order);
    }

  public:
    void
    setup_unknowns()
    {
      TimerOutput::Scope timer(timer_output, "Setup unknowns + operator");
      dof_handler.reinit(triangulation);
      dof_handler.distribute_dofs(fe);
      locally_owned_dofs = dof_handler.locally_owned_dofs();

      morton_reorder_dofs();

      // for the solver, we use hp-multigrid according to
      // https://doi.org/10.1016/j.jcp.2020.109538 and
      // https://doi.org/10.1145/3580314

      // start by creating levels of continuous elements
      std::vector<unsigned int> p_levels({fe.degree + 1});
      while (p_levels.back() > 2)
        p_levels.push_back(std::max(p_levels.back() - 2, 2u));
      fes_solve.resize(0, p_levels.size() - 1);
      for (unsigned int level = 0; level < p_levels.size(); ++level)
        fes_solve[level] =
          std::make_unique<FE_Q<dim>>(p_levels[p_levels.size() - 1 - level]);

      coarse_triangulations =
        MGTransferGlobalCoarseningTools::create_geometric_coarsening_sequence(
          triangulation);
      dof_handlers_solve.resize(0,
                                coarse_triangulations.size() - 1 +
                                  fes_solve.max_level());
      constraints_solve.resize(0, dof_handlers_solve.max_level());
      for (unsigned int level = dof_handlers_solve.min_level();
           level <= dof_handlers_solve.max_level();
           ++level)
        {
          DoFHandler<dim> &dof_h = dof_handlers_solve[level];
          dof_h.reinit(*coarse_triangulations[std::min(
            level, triangulation.n_global_levels() - 1)]);
          if (level < coarse_triangulations.size())
            dof_h.distribute_dofs(*fes_solve[0]);
          else
            dof_h.distribute_dofs(
              *fes_solve[level + 1 - coarse_triangulations.size()]);

          IndexSet relevant_dofs =
          DoFTools::extract_locally_relevant_dofs(dof_h);
          AffineConstraints<float> &level_constraints =
            constraints_solve[level];
          level_constraints.reinit(dof_h.locally_owned_dofs(), relevant_dofs);
          std::vector<GridTools::PeriodicFacePair<
            typename DoFHandler<dim>::cell_iterator>>
            periodic_faces;
          for (unsigned int d = 0; d < dim; ++d)
            GridTools::collect_periodic_faces(
              dof_h, 2 * d, 2 * d + 1, d, periodic_faces);
          DoFTools::make_periodicity_constraints<dim, dim>(periodic_faces,
                                                           level_constraints);
          DoFTools::make_hanging_node_constraints(dof_h, level_constraints);
          level_constraints.close();
          typename MatrixFree<dim, float>::AdditionalData additional_data;
          additional_data.tasks_parallel_scheme =
            MatrixFree<dim, float>::AdditionalData::none;

          DoFRenumbering::matrix_free_data_locality(dof_h,
                                                    level_constraints,
                                                    additional_data);

          // with the new numbering, create the final data structures
          relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_h);
          level_constraints.clear();
          level_constraints.reinit(dof_h.locally_owned_dofs(), relevant_dofs);
          DoFTools::make_periodicity_constraints<dim, dim>(periodic_faces,
                                                           level_constraints);
          DoFTools::make_hanging_node_constraints(dof_h, level_constraints);
          level_constraints.close();
        }

      AffineConstraints<double> constraints_fine;
      constraints_fine.reinit(dof_handlers_solve.back().locally_owned_dofs(),
                              constraints_solve.back().get_local_lines());
      constraints_fine.copy_from(constraints_solve.back());
      fine_matrix.initialize(mapping,
                             dof_handlers_solve.back(),
                             constraints_fine,
                             fes_solve.back()->degree + 1,
                             dof_handler);
      fine_matrix.initialize_dof_vector(vector_f, 1);
      fine_matrix.initialize_dof_vector(vector_u, 1);
      fine_matrix.initialize_dof_vector(rhs, 0);
      solution.resize(4);
      for (auto &sol : solution)
        fine_matrix.initialize_dof_vector(sol, 0);
      matvec_solutions.resize(4);
      for (auto &sol : matvec_solutions)
        fine_matrix.initialize_dof_vector(sol, 0);
      VectorTools::interpolate(dof_handler, InitNaN<dim>(), vector_f);
    }

    void
    setup_multigrid_operators_and_transfer()
    {
      TimerOutput::Scope timer(timer_output, "Setup multigrid");
      level_matrices.resize(0, dof_handlers_solve.max_level());
      mg_transfers.resize(0, dof_handlers_solve.max_level());
      MGLevelObject<typename SmootherType::AdditionalData> smoother_data(
        0, dof_handlers_solve.max_level());
      for (unsigned int level = dof_handlers_solve.min_level();
           level <= dof_handlers_solve.max_level();
           ++level)
        {
          level_matrices[level].initialize(
            mapping,
            dof_handlers_solve[level],
            constraints_solve[level],
            dof_handlers_solve[level].get_fe().degree + 1);
          level_matrices[level].compute_inverse_diagonal();

          // manually compute the eigenvalue estimate for Chebyshev because we
          // need to be careful with the constrained indices
          IterationNumberControl control(12, 1e-6, false, false);

          SolverCG<VectorTypef> solver(control);
          dealii::internal::EigenvalueTracker
            eigenvalue_tracker;
          solver.connect_eigenvalues_slot(
            [&eigenvalue_tracker](const std::vector<double> &eigenvalues) {
              eigenvalue_tracker.slot(eigenvalues);
            });

          VectorTypef sol, rhs;
          level_matrices[level].initialize_dof_vector(sol);
          level_matrices[level].initialize_dof_vector(rhs);

          for (unsigned int i = 0; i < rhs.locally_owned_size(); ++i)
            rhs.local_element(i) = 0.5 * (i % 2) + 0.2 * (i % 4) +
                                   0.1 * (i % 8) + 0.05 * (i % 16) +
                                   0.025 * (i % 32) + 0.0125 * (i % 64);
          make_zero_mean(
            level_matrices[level].get_matrix_free().get_constrained_dofs(),
            rhs);
          solver.solve(level_matrices[level],
                       sol,
                       rhs,
                       *level_matrices[level].get_matrix_diagonal_inverse());

          if (level > 0)
            {
              smoother_data[level].smoothing_range = 15.;
              smoother_data[level].degree          = 4;
            }
          else
            {
              // Coarse level: Use MG smoother as solver (should use p-multigrid
              // or AMG for complicated meshes)
              smoother_data[level].smoothing_range =
                eigenvalue_tracker.values.back() /
                eigenvalue_tracker.values.front();
              smoother_data[0].degree = numbers::invalid_unsigned_int;
            }
          smoother_data[level].max_eigenvalue =
            eigenvalue_tracker.values.back();
          smoother_data[level].eig_cg_n_iterations = 0;
          smoother_data[level].preconditioner =
            level_matrices[level].get_matrix_diagonal_inverse();
          if (level > 0)
            mg_transfers[level].reinit(dof_handlers_solve[level],
                                       dof_handlers_solve[level - 1],
                                       constraints_solve[level],
                                       constraints_solve[level - 1]);
        }
      mg_smoother.initialize(level_matrices, smoother_data);

      mg_transfer =
        std::make_unique<MGTransferGlobalCoarsening<dim, VectorTypeMG>>(
          mg_transfers, [&](const unsigned level, VectorTypeMG &vec) {
            level_matrices[level].initialize_dof_vector(vec);
          });

      // resize AMR indicator storage to match current number of cells
      vector_amr_indicator.reinit(triangulation.n_locally_owned_active_cells());
    }

    void
    print_timings() const override
    {
      int mpi_has_been_finalized = 0;
      MPI_Finalized(&mpi_has_been_finalized);
      if (mpi_has_been_finalized == 0 && dof_handlers_solve.back().n_dofs() > 0)
        {
          pcout << "Number of DoFs: " << dof_handlers_solve.back().n_dofs()
                << std::endl;
          pcout << "Throughput Poisson solver: "
                << dof_handlers_solve.back().n_dofs() *
                     timer_output.get_summary_data(
                       TimerOutput::n_calls)["Solve run solver"] /
                     timer_output.get_summary_data(
                       TimerOutput::total_wall_time)["Solve run solver"] *
                     1e-6
                << " MDoF/sec" << std::endl;
          timer_output.print_wall_time_statistics(MPI_COMM_WORLD);
        }
    }

    void
    solve() override
    {
      timer_output.enter_subsection("Solve rhs + mg precondition");

      // Prepare right hand side
      {
        rhs = 0.;
        FEEvaluation<dim, -1, 0, 1, double> eval_low(
          fine_matrix.get_matrix_free(), 1);
        FEEvaluation<dim, -1, 0, 1, double> eval(fine_matrix.get_matrix_free(),
                                                 0);
        for (unsigned int cell = 0;
             cell < fine_matrix.get_matrix_free().n_cell_batches();
             ++cell)
          {
            eval.reinit(cell);
            eval_low.reinit(cell);
            eval_low.gather_evaluate(vector_f, EvaluationFlags::values);
            for (unsigned int q = 0; q < eval.n_q_points; ++q)
              eval.submit_value(eval_low.get_value(q), q);
            eval.integrate_scatter(EvaluationFlags::values, rhs);
          }
        rhs.compress(VectorOperation::add);

        // Since we use periodicity in all directions, we need to make the
        // right hand side free of a mean value
        make_zero_mean(fine_matrix.get_matrix_free().get_constrained_dofs(),
                       rhs);
      }

      // Adjust the mean value due to periodicity inside the coarse solver:
      // This is not strictly necessary as we will fix the global vectors, but
      // the singularity in the coarse solver might shift the solution and
      // lead to large values (and hence more roundoff) if we do not adjust it
      // there
      MGCoarseSolverPeriodicity<VectorTypeMG> mg_coarse;
      mg_coarse.initialize(
        mg_smoother,
        level_matrices[0].get_matrix_free().get_constrained_dofs(),
        true);
      mg::Matrix<VectorTypeMG> mg_matrix(level_matrices);

      Multigrid<VectorTypeMG> mg(
        mg_matrix, mg_coarse, *mg_transfer, mg_smoother, mg_smoother);
      PreconditionMG<dim,
                     VectorTypeMG,
                     MGTransferGlobalCoarsening<dim, VectorTypeMG>>
        preconditioner(dof_handlers_solve.back(), mg, *mg_transfer);

      timer_output.leave_subsection("Solve rhs + mg precondition");

      {
        TimerOutput::Scope timer(timer_output,
                                 "Solve project from old solutions");
        const unsigned int n_max_steps = solution.size();
        FullMatrix<double> projection_matrix(n_max_steps, n_max_steps);
        unsigned int       step = 0;
        for (; step < n_max_steps; ++step)
          {
            fine_matrix.vmult(matvec_solutions[step], solution[step]);
            // Modified Gram-Schmidt
            projection_matrix(0, step) =
              matvec_solutions[step] * matvec_solutions[0];
            for (unsigned int j = 0; j < step; ++j)
              projection_matrix(j + 1, step) =
                matvec_solutions[step].add_and_dot(-projection_matrix(j, step) /
                                                     projection_matrix(j, j),
                                                   matvec_solutions[j],
                                                   matvec_solutions[j + 1]);

            // Note that the entries in the matrix are the square of the norm,
            // so we request that vectors which are below 1e-8 in terms of
            // orthogonality are discarded (almost linearly dependent)
            if (projection_matrix(0, 0) == 0 ||
                projection_matrix(step, step) < 1e-16 * projection_matrix(0, 0))
              break;
          }
        // Solve least-squares system
        std::array<double, 4> project_sol = {};
        for (unsigned int s = 0; s < step; ++s)
          project_sol[s] = matvec_solutions[s] * rhs;
        for (int s = step - 1; s >= 0; --s)
          {
            double sum = project_sol[s];
            for (unsigned int j = s + 1; j < step; ++j)
              sum -= project_sol[j] * projection_matrix(s, j);
            project_sol[s] = sum / projection_matrix(s, s);
          }

        // extrapolate solution from old values and write into last vector
        const unsigned int local_size = solution[0].locally_owned_size();
        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0; i < local_size; ++i)
          {
            const double sol_0 = solution[0].local_element(i);
            const double sol_1 = solution[1].local_element(i);
            const double sol_2 = solution[2].local_element(i);
            const double sol_3 = solution[3].local_element(i);
            solution[3].local_element(i) =
              project_sol[0] * sol_0 + project_sol[1] * sol_1 +
              project_sol[2] * sol_2 + project_sol[3] * sol_3;
          }

        // rotate old solutions back by one step, moving the extrapolated
        // solution to the front
        solution[3].swap(solution[2]);
        solution[2].swap(solution[1]);
        solution[1].swap(solution[0]);
      }

      {
        TimerOutput::Scope timer(timer_output, "Solve run solver");
        SolverControl      control(maxiters, abstol * rhs.l2_norm());
        SolverCG<VectorTyped> solver(control);

        solver.solve(fine_matrix, solution[0], rhs, preconditioner);

        // The solution should have zero mean after solving, but roundoff
        // errors might create a shift (note that we do multigrid in single
        // precision, so the coarse solver's adjustment is only accurate down
        // to 1e-7 or so)
        make_zero_mean(fine_matrix.get_matrix_free().get_constrained_dofs(),
                       solution[0]);

#ifdef DEBUG_VERBOSE
        if (!Utilities::MPI::job_supports_mpi() ||
            Utilities::MPI::this_mpi_process(
              triangulation.get_communicator()) == 0)
          std::cout << ">>> Conjugate gradient solver for MG converged in "
                    << control.last_step() << " iterations, norms "
                    << control.initial_value() << " " << control.last_value()
                    << std::endl;
#endif
      }

      TimerOutput::Scope timer(timer_output, "Solve u project");
      // project solution to low degree
      fine_matrix.get_matrix_free().template cell_loop<VectorTyped, VectorTyped>(
        [](const MatrixFree<dim, double> &              data,
           VectorTyped &                                dst,
           const VectorTyped &                          src,
           const std::pair<unsigned int, unsigned int> &cell_range) {
          FEEvaluation<dim, -1, 0, 1> eval(data, 0, 1);
          FEEvaluation<dim, -1, 0, 1> eval_low(data, 1, 1);
          MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, 1>
            inverse_mass(eval_low);
          for (unsigned int cell = cell_range.first; cell < cell_range.second;
               ++cell)
            {
              eval.reinit(cell);
              eval_low.reinit(cell);
              eval.gather_evaluate(src, EvaluationFlags::values);
              inverse_mass.transform_from_q_points_to_basis(
                1, eval.begin_values(), eval_low.begin_dof_values());
              eval_low.set_dof_values(dst);
            }
        },
        vector_u,
        solution[0]);
    }

    template <typename IteratorType, typename FunctionType>
    void
    process_cells_recursively_on_tree(const IteratorType &cell,
                                      const FunctionType &operation_on_cell)
    {
      if (cell->has_children())
        for (unsigned int child = 0; child < cell->n_children(); ++child)
          process_cells_recursively_on_tree(cell->child(child),
                                            operation_on_cell);
      else if (cell->is_locally_owned()) // \equiv !is_ghost() && !is_artificial()
        operation_on_cell(cell);
    }

    char filename[64] = "mesh_output/grid_out.x.0000.svg";
    int kth_out = 0;
    GridOut grid_out = GridOut();
    constexpr static bool gridoutput = true;

    void
    output_grid() override
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (dim == 2)
          {
            sprintf(filename, "mesh_output/grid_out.%d.%04d.svg", rank, kth_out++);
            std::ofstream gridstream = std::ofstream(filename);
            for (const auto &coarse_cell : triangulation.cell_iterators_on_level(0)) {
              process_cells_recursively_on_tree(coarse_cell, [&](auto &cell) {
                gridstream << cell->barycenter() << " " << bool(cell->refine_flag_set()) << " "
                           << cell->coarsen_flag_set() << "\n";
              });
            }
            //GridOutFlags::Svg svg_flags;
            //svg_flags.coloring = GridOutFlags::Svg::Coloring::subdomain_id;
            ////svg_flags.coloring = GridOutFlags::Svg::Coloring::level_number;
            //svg_flags.background = GridOutFlags::Svg::Background::transparent;
            //svg_flags.label_cell_index = true;
            //svg_flags.label_subdomain_id = true;
            //grid_out.set_flags(svg_flags);
            //grid_out.write_svg(triangulation, gridstream);
          }
        else
          {
            sprintf(filename, "grid_out.%d.%03d.vtu", rank, kth_out++);
            std::ofstream gridstream = std::ofstream(filename);
            grid_out.write_vtu(triangulation, gridstream);
          }
    }

    void
    adapt_grid() override
    {
      TimerOutput::Scope timer(timer_output, "Refine mesh");

      // Mark cells for refinement/coarsening according to the given vector
      // (1 = refine, 0 no action, -1 coarsen)
      // Run a manual loop to ensure the Morton order used by p4est and trixi
      long counter = 0, num_cells = 0;
      int will_adapt = false;
      for (const auto &coarse_cell : triangulation.cell_iterators_on_level(0))
        process_cells_recursively_on_tree(coarse_cell, [&](auto &cell) {
          const int indicator = vector_amr_indicator[counter++];
          if (indicator == 1) {
            cell->set_refine_flag();
            will_adapt = true;
          } else if (indicator == -1) {
            cell->set_coarsen_flag();
            will_adapt = true;
          }
        });
      MPI_Allreduce(MPI_IN_PLACE, &will_adapt, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);

      if constexpr (gridoutput)
         output_grid();

      // actually perform adaptation
      if (will_adapt) {
        num_cells = triangulation.n_locally_owned_active_cells();
        MPI_Allreduce(MPI_IN_PLACE, &counter, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, &num_cells, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (!rank) {
           std::cerr << "pre-adaption " << num_cells << " ; iterator size " << counter << "\n";
        }

        triangulation.execute_coarsening_and_refinement();

        num_cells = triangulation.n_locally_owned_active_cells();
        MPI_Allreduce(MPI_IN_PLACE, &num_cells, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
        if (!rank)
           std::cerr << "post-adaption " << num_cells << " iteration number: " << kth_out << "\n";
      }

      // update any other data structures
      if (will_adapt) {
        setup_unknowns();
        setup_multigrid_operators_and_transfer();
      }
    }

    // repartitions the deal.II internal grid on the MPI ranks
    void
    repartition_grid() override
    {
      triangulation.repartition();
    }

    unsigned int
    get_num_local_active_cells() const override
    {
      return triangulation.n_locally_owned_active_cells();
    }

    unsigned int
    get_mesh_checksum() const override
    {
      return triangulation.get_checksum();
    }

    std::array<double *, 3>
    compute_grad_u() override
    {
      TimerOutput::Scope timer(timer_output, "Compute gradient of u");
      vector_grad_u.reinit(dim);
      for (unsigned int d = 0; d < dim; ++d)
        fine_matrix.initialize_dof_vector(vector_grad_u.block(d), 1);
      vector_grad_u.collect_sizes();

      using BlockVectorType = LinearAlgebra::distributed::BlockVector<double>;
      // this loop implements (q, nabla u) + <q, ({{u}} - u) n> =
      // = (q, nabla u) - <q, 0.5 * [[u]]>
      fine_matrix.get_matrix_free().template loop<BlockVectorType, VectorTyped>(
        [](const MatrixFree<dim, double> &              data,
           BlockVectorType &                            dst,
           const VectorTyped &                          src,
           const std::pair<unsigned int, unsigned int> &cell_range) {
          FEEvaluation<dim, -1, 0, 1>   eval_scalar(data, 0);
          FEEvaluation<dim, -1, 0, dim> eval_vector(data, 1);
          for (unsigned int cell = cell_range.first; cell < cell_range.second;
               ++cell)
            {
              eval_scalar.reinit(cell);
              eval_vector.reinit(cell);
              eval_scalar.gather_evaluate(src, EvaluationFlags::gradients);
              for (unsigned int q = 0; q < eval_scalar.n_q_points; ++q)
                eval_vector.submit_value(eval_scalar.get_gradient(q), q);
              eval_vector.integrate_scatter(EvaluationFlags::values, dst);
            }
        },
        [](const MatrixFree<dim, double> &              data,
           BlockVectorType &                            dst,
           const VectorTyped &                          src,
           const std::pair<unsigned int, unsigned int> &face_range) {
          FEFaceEvaluation<dim, -1, 0, 1>   eval_scalar_m(data, true, 0);
          FEFaceEvaluation<dim, -1, 0, 1>   eval_scalar_p(data, true, 0);
          FEFaceEvaluation<dim, -1, 0, dim> eval_vector_m(data, true, 1);
          FEFaceEvaluation<dim, -1, 0, dim> eval_vector_p(data, true, 1);
          for (unsigned int face = face_range.first; face < face_range.second;
               ++face)
            {
              eval_scalar_m.reinit(face);
              eval_vector_m.reinit(face);
              eval_scalar_p.reinit(face);
              eval_vector_p.reinit(face);
              eval_scalar_m.gather_evaluate(src, EvaluationFlags::values);
              eval_scalar_p.gather_evaluate(src, EvaluationFlags::values);
              for (unsigned int q = 0; q < eval_scalar_m.n_q_points; ++q)
                {
                  const auto jump =
                    eval_scalar_m.get_value(q) - eval_scalar_p.get_value(q);
                  const auto normal = eval_scalar_m.normal_vector(q);
                  eval_vector_m.submit_value(normal * (-0.5 * jump), q);
                  eval_vector_p.submit_value(normal * (0.5 * jump), q);
                }
              eval_vector_m.integrate_scatter(EvaluationFlags::values, dst);
              eval_vector_p.integrate_scatter(EvaluationFlags::values, dst);
            }
        },
        [](const MatrixFree<dim, double> &,
           BlockVectorType &,
           const VectorTyped &,
           const std::pair<unsigned int, unsigned int> &) {
          AssertThrow(
            false, ExcMessage("Boundary operation currently not implemented"));
        },
        vector_grad_u,
        solution[0],
        true);

      // apply inverse mass matrix in-place on the vector_grad_u vector
      fine_matrix.get_matrix_free()
        .template cell_loop<BlockVectorType, BlockVectorType>(
          [](const MatrixFree<dim, double> &data,
             BlockVectorType &              dst,
             const BlockVectorType &,
             const std::pair<unsigned int, unsigned int> &cell_range) {
            FEEvaluation<dim, -1, 0, dim> eval_vector(data, 1, 1);
            MatrixFreeOperators::CellwiseInverseMassMatrix<dim, -1, dim>
              inverse_mass(eval_vector);
            for (unsigned int cell = cell_range.first; cell < cell_range.second;
                 ++cell)
              {
                eval_vector.reinit(cell);
                eval_vector.read_dof_values(dst);
                inverse_mass.apply(eval_vector.begin_dof_values(),
                                   eval_vector.begin_dof_values());
                eval_vector.set_dof_values(dst);
              }
          },
          vector_grad_u,
          vector_grad_u);

      std::array<double *, 3> grad_u;
      for (unsigned int d = 0; d < dim; ++d)
        grad_u[d] = vector_grad_u.block(d).begin();
      for (unsigned int d = dim; d < 3; ++d)
        grad_u[d] = nullptr;
      return grad_u;
    }

    parallel::distributed::Triangulation<dim> triangulation;
    std::vector<std::shared_ptr<const Triangulation<dim>>>
                                              coarse_triangulations;
    MappingQ<dim>                             mapping;
    FE_DGQ<dim>                               fe;
    DoFHandler<dim>                           dof_handler;
    MGLevelObject<std::unique_ptr<FE_Q<dim>>> fes_solve;
    MGLevelObject<DoFHandler<dim>>            dof_handlers_solve;

    VectorTyped rhs; // integrated source term of the Poisson problem (high degree)
    std::vector<VectorTyped> solution; // solutions of the Poisson problem (high degree)
    std::vector<VectorTyped> matvec_solutions; // matrix-vector product of old solutions used
                                               // during projection
    LinearAlgebra::distributed::BlockVector<double>
      vector_grad_u; // gradient of the solution to the Poisson problem

    LaplaceOperator<dim, double>               fine_matrix;
    MGLevelObject<AffineConstraints<float>>    constraints_solve;
    MGLevelObject<LaplaceOperator<dim, float>> level_matrices;
    using VectorTypeMG = LinearAlgebra::distributed::Vector<float>;

    using SmootherType =
      PreconditionChebyshev<LaplaceOperator<dim, float>, VectorTypeMG>;
    mg::SmootherRelaxation<SmootherType, VectorTypeMG> mg_smoother;

    MGLevelObject<MGTwoLevelTransfer<dim, VectorTypeMG>>           mg_transfers;
    std::unique_ptr<MGTransferGlobalCoarsening<dim, VectorTypeMG>> mg_transfer;

    ConditionalOStream pcout;
    TimerOutput        timer_output;
  };
} // namespace PoissonModule



extern "C"
{
  void *
  init_libs(int argc, char **argv)
  {
#ifdef DEBUG_VERBOSE
    std::cout << ">>> libdealii: init_libs" << std::endl;
#endif
    return new dealii::InitFinalize(argc, argv,
                                    dealii::InitializeLibrary::Kokkos |
                                    dealii::InitializeLibrary::SLEPc |
                                    dealii::InitializeLibrary::PETSc |
                                    dealii::InitializeLibrary::Zoltan,
                                    1);
  }

  void *
  init_2d(int polydeg, MeshSettings2D meshsettings, int maxiters, double abstol)
  {
    PoissonModule::WrapperBase *wrapper = new PoissonModule::Wrapper<2>(polydeg, meshsettings, maxiters, abstol);
    return wrapper;
  }

  void *
  init_3d(int polydeg, MeshSettings3D meshsettings, int maxiters, double abstol)
  {
    PoissonModule::WrapperBase *wrapper = new PoissonModule::Wrapper<3>(polydeg, meshsettings, maxiters, abstol);
    return wrapper;
  }

  void
  set_maxiters(void *wrapped, int maxiters)
  {
    auto &object    = *static_cast<PoissonModule::WrapperBase *>(wrapped);
    object.maxiters = maxiters;
  }

  void
  set_abstol(void *wrapped, double abstol)
  {
    auto &object  = *static_cast<PoissonModule::WrapperBase *>(wrapped);
    object.abstol = abstol;
  }

  unsigned long
  get_problem_size_on_rank(void *wrapped)
  {
    auto &object = *static_cast<PoissonModule::WrapperBase *>(wrapped);
    return object.locally_owned_dofs.n_elements();
  }

  double *
  get_pointer_f(void *wrapped)
  {
    auto &object = *static_cast<PoissonModule::WrapperBase *>(wrapped);
    return object.vector_f.begin();
  }

  double *
  get_pointer_u(void *wrapped)
  {
    auto &object = *static_cast<PoissonModule::WrapperBase *>(wrapped);
    return object.vector_u.begin();
  }

  double **
  get_pointer_grad_u(void *wrapped)
  {
    auto &object = *static_cast<PoissonModule::WrapperBase *>(wrapped);
    object.pointers_grad_u = object.compute_grad_u();
    return object.pointers_grad_u.data();
  }

  int *
  get_pointer_amr_indicator(void *wrapped)
  {
    auto &object = *static_cast<PoissonModule::WrapperBase *>(wrapped);
    return object.vector_amr_indicator.begin();
  }

  void
  solve(void *wrapped)
  {
    auto &object = *static_cast<PoissonModule::WrapperBase *>(wrapped);
    object.solve();
  }

  void
  adapt_grid(void *wrapped)
  {
    auto &object = *static_cast<PoissonModule::WrapperBase *>(wrapped);
    object.adapt_grid();
  }

  void
  repartition_grid(void *wrapped)
  {
     auto& object = *static_cast<PoissonModule::WrapperBase *>(wrapped);
     object.repartition_grid();
  }

  unsigned int
  get_num_local_active_cells(void *wrapped)
  {
     auto& object = *static_cast<PoissonModule::WrapperBase *>(wrapped);
     return object.get_num_local_active_cells();
  }

  unsigned int
  get_mesh_checksum(void *wrapped)
  {
    auto &object = *static_cast<PoissonModule::WrapperBase *>(wrapped);
    return object.get_mesh_checksum();
  }

  void
  finalize(void *wrapped)
  {
    auto *object = static_cast<PoissonModule::WrapperBase *>(wrapped);
    object->print_timings();
    delete object;
  }

  void
  finalize_libs(void *mpi)
  {
    auto *object = static_cast<dealii::Utilities::MPI::MPI_InitFinalize *>(mpi);
    delete object;
  }
}
