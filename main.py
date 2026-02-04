import argparse
import sys
from tasks.manifold import run_manifold_task
from tasks.hyperbolic import run_hyperbolic_task
from tasks.semantic import run_semantic_task
from tasks.cross_modal import run_crossmodal_task
from tasks.sanity_check import run_sanity_check

def main():
    parser = argparse.ArgumentParser(description="Versor: Geometric Algebra Neural Networks CLI")
    
    subparsers = parser.add_subparsers(dest='task', help='Available tasks')
    
    # Task: Manifold Restoration
    parser_manifold = subparsers.add_parser('manifold', help='Run Figure-8 Manifold Restoration (Unbending)')
    parser_manifold.add_argument('--epochs', type=int, default=800, help='Number of training epochs')
    parser_manifold.add_argument('--lr', type=float, default=0.02, help='Learning rate')

    # Task: Hyperbolic Geometry
    parser_hyperbolic = subparsers.add_parser('hyperbolic', help='Run Hyperbolic Geometry Task (Lorentz Boost)')
    parser_hyperbolic.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser_hyperbolic.add_argument('--lr', type=float, default=0.05, help='Learning rate')

    # Task: Semantic Unbending (BERT)
    parser_semantic = subparsers.add_parser('semantic', help='Run Semantic Manifold Unbending (BERT Projection)')
    parser_semantic.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser_semantic.add_argument('--lr', type=float, default=0.01, help='Learning rate')

    # Task: Cross-Modal Unification
    parser_cross = subparsers.add_parser('crossmodal', help='Run Cross-Modal Manifold Unification')
    parser_cross.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser_cross.add_argument('--lr', type=float, default=0.01, help='Learning rate')

    # Task: Sanity Check (Random Noise)
    parser_sanity = subparsers.add_parser('sanity', help='Run Sanity Check (Random Noise Input)')
    parser_sanity.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser_sanity.add_argument('--lr', type=float, default=0.01, help='Learning rate')

    args = parser.parse_args()
    
    if args.task == 'manifold':
        run_manifold_task(epochs=args.epochs, lr=args.lr)
    elif args.task == 'hyperbolic':
        run_hyperbolic_task(epochs=args.epochs, lr=args.lr)
    elif args.task == 'semantic':
        run_semantic_task(epochs=args.epochs, lr=args.lr)
    elif args.task == 'crossmodal':
        run_crossmodal_task(epochs=args.epochs, lr=args.lr)
    elif args.task == 'sanity':
        run_sanity_check(epochs=args.epochs, lr=args.lr)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()