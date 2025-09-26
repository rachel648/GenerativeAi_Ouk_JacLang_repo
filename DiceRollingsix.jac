"""Scale-Agnostic Dice Rolling Game with LLM"""

walker DiceGame {
    has roll: int;

    # Start the game
    can start with entry {
        print("🎲 Starting Dice Game...");
        self.roll_dice();
    }

    # Roll dice and optionally call AI
    can roll_dice with entry {
        set self.roll = rand(1,6);
        print(f"You rolled: {self.roll}");

        # AI hint (will work in a cloud runtime)
        let hint = give_hint(self.roll);
        print(f"💡 Hint: {hint}");
    }
}

"""LLM hint function (scale-agnostic)"""
def give_hint(roll: int) -> str by llm();

# Entry point
with entry {
    root.spawn DiceGame();
    root.spawn DiceGame();
    root.spawn DiceGame();
    root.spawn DiceGame();
}
