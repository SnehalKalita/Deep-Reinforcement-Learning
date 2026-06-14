import numpy as np
import matplotlib.pyplot as plt

def generate_phase_portrait():
    # 1. Setup the time array (simulating 15 seconds of physics)
    t = np.linspace(0, 15, 1000)

    # 2. Scenario 1: Expert Policy (Stable Limit Cycle)
    # The expert repeats the same closed loop endlessly (sine/cosine)
    angle_expert = np.sin(t)
    velocity_expert = np.cos(t)

    # 3. Scenario 2: Symbolic Policy (Diverging Spiral)
    # The math drops contact forces, causing an exponential divergence every step
    growth_rate = 0.15
    angle_symbolic = np.exp(growth_rate * t) * np.sin(t)
    velocity_symbolic = np.exp(growth_rate * t) * np.cos(t)

    # 4. Initialize the plot
    # A square figure size (8x8) is standard for phase portraits so the axes are scaled equally
    plt.figure(figsize=(8, 8))

    # Plot the expert policy (thick solid blue line)
    plt.plot(angle_expert, velocity_expert, 
             label='Expert Policy (Stable Limit Cycle)', 
             color='#1f77b4', linewidth=3)

    # Plot the symbolic policy (dashed red line)
    plt.plot(angle_symbolic, velocity_symbolic, 
             label='Symbolic Policy (Diverging Spiral)', 
             color='#d62728', linestyle='--', linewidth=2.5)

    # 5. Styling for a professional report
    # Add origin axes
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    
    # Add a subtle grid
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Labels and Title
    plt.xlabel('Joint Angle [rad]', fontsize=12, fontweight='bold')
    plt.ylabel('Joint Angular Velocity [rad/s]', fontsize=12, fontweight='bold')
    plt.title('Phase Portrait: Control Mechanics Divergence\nExpert vs. Symbolic Policy', 
              fontsize=14, fontweight='bold', pad=15)
    
    # Legend
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)

    # Lock the axis limits so the spiral doesn't dwarf the limit cycle
    plt.xlim(-8, 8)
    plt.ylim(-8, 8)

    # 6. Save the output
    # dpi=300 ensures it is print-quality for a PDF report
    output_filename = 'phase_portrait_comparison.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Graph successfully saved as: {output_filename}")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    generate_phase_portrait()