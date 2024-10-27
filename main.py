import numpy as np
import sqlite3
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


class SelfLearningChatbot:
    def __init__(self, db_path="chatbot.db", confidence_threshold=0.4):
        """Initialize the self-learning chatbot with existing SQLite database."""
        self.db_path = db_path
        self.confidence_threshold = confidence_threshold
        self.conversation_history = []
        self._init_database()
        self._prepare_training_data()
        self._train_model()

    def _init_database(self):
        """Initialize the database with learning-related tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create feedback table with correct schema
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT NOT NULL,
                    bot_response TEXT NOT NULL,
                    intent_tag TEXT,
                    feedback_score INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create learned patterns table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    intent_id INTEGER,
                    pattern TEXT NOT NULL,
                    confidence FLOAT DEFAULT 1.0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (intent_id) REFERENCES intents (id)
                )
            """
            )

            conn.commit()

    def _prepare_training_data(self):
        """Prepare training data from both original and learned patterns."""
        self.X = []
        self.y = []

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Get patterns from original patterns table
                cursor.execute(
                    """
                    SELECT p.pattern, i.tag 
                    FROM patterns p 
                    JOIN intents i ON p.intent_id = i.id
                """
                )

                for pattern, tag in cursor.fetchall():
                    self.X.append(pattern.lower())
                    self.y.append(tag)

                # Get patterns from learned patterns table
                cursor.execute(
                    """
                    SELECT lp.pattern, i.tag 
                    FROM learned_patterns lp 
                    JOIN intents i ON lp.intent_id = i.id
                """
                )

                for pattern, tag in cursor.fetchall():
                    self.X.append(pattern.lower())
                    self.y.append(tag)

        except Exception as e:
            print(f"Error preparing training data: {e}")
            raise

    def _train_model(self):
        """Train the model with available data."""
        if not self.X or not self.y:
            raise ValueError("No training data available")

        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=100, random_state=42),
                ),
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.pipeline.fit(X_train, y_train)
        self.accuracy = self.pipeline.score(X_test, y_test)
        print(f"Model trained with accuracy: {self.accuracy:.2f}")

    def learn_from_interaction(
        self, user_input, response, feedback_score, intent_tag=None
    ):
        """Learn from user interactions with corrected schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Store feedback with correct column names
                cursor.execute(
                    """
                    INSERT INTO user_feedback 
                    (user_input, bot_response, intent_tag, feedback_score) 
                    VALUES (?, ?, ?, ?)
                """,
                    (user_input, response, intent_tag, feedback_score),
                )

                # If feedback is positive, consider adding to learned patterns
                if feedback_score > 3:  # Changed to match 1-5 scale
                    predicted_tag, _ = self.predict_intent(user_input)

                    if predicted_tag:
                        cursor.execute(
                            "SELECT id FROM intents WHERE tag = ?", (predicted_tag,)
                        )
                        intent_id = cursor.fetchone()[0]

                        # Add to learned patterns if not exists
                        cursor.execute(
                            """
                            INSERT INTO learned_patterns (intent_id, pattern, confidence)
                            SELECT ?, ?, ?
                            WHERE NOT EXISTS (
                                SELECT 1 FROM learned_patterns 
                                WHERE pattern = ? AND intent_id = ?
                            )
                        """,
                            (
                                intent_id,
                                user_input,
                                feedback_score / 5,
                                user_input,
                                intent_id,
                            ),
                        )

                conn.commit()

            # Check if retraining is needed
            self._check_and_retrain()
            return True

        except Exception as e:
            print(f"Error in learning process: {e}")
            return False

    def _check_and_retrain(self):
        """Check if retraining is needed based on new learned patterns."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM learned_patterns")
            learned_count = cursor.fetchone()[0]

            # Retrain if we have enough new patterns (arbitrary threshold)
            if learned_count % 5 == 0 and learned_count > 0:
                self._prepare_training_data()
                self._train_model()

    def predict_intent(self, user_input):
        """Predict intent with confidence score."""
        processed_input = user_input.lower()
        predicted_tag = self.pipeline.predict([processed_input])[0]
        confidence = np.max(self.pipeline.predict_proba([processed_input]))
        return (
            predicted_tag if confidence >= self.confidence_threshold else None,
            confidence,
        )

    def get_response(self, tag):
        """Get response for the predicted tag."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT response 
                    FROM responses r
                    JOIN intents i ON r.intent_id = i.id
                    WHERE i.tag = ?
                """,
                    (tag,),
                )

                responses = cursor.fetchall()
                if responses:
                    return np.random.choice([r[0] for r in responses])
                return "I'm still learning about that."

        except Exception as e:
            print(f"Error getting response: {e}")
            return "I encountered an error while processing your request."

    def chat(self, user_input):
        """Main chat function with learning capabilities."""
        predicted_tag, confidence = self.predict_intent(user_input)

        if predicted_tag is None:
            response = (
                "I'm not quite sure about that yet. Could you rephrase or teach me?"
            )
            feedback_score = 0
        else:
            response = self.get_response(predicted_tag)
            feedback_score = int(confidence * 5)  # Convert confidence to 1-5 scale

        # Learn from this interaction
        self.learn_from_interaction(user_input, response, feedback_score, predicted_tag)

        return response, confidence


def main():
    """Main function to run the chatbot."""
    print("Initializing self-learning chatbot...")
    try:
        chatbot = SelfLearningChatbot()
        print("Chatbot is ready! Type 'exit' to end the chat.")

        while True:
            user_input = input("You: ").strip()

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Chatbot: Goodbye! Thanks for helping me learn!")
                break

            response, confidence = chatbot.chat(user_input)
            print(f"Chatbot: {response}")
            print(f"Confidence: {confidence:.2f}")

            # Optional: Collect explicit feedback
            feedback = input(
                "Was this response helpful? (1-5, or press Enter to skip): "
            ).strip()
            if feedback.isdigit() and 1 <= int(feedback) <= 5:
                chatbot.learn_from_interaction(
                    user_input, response, int(feedback), None
                )

    except KeyboardInterrupt:
        print("\nChatbot: Goodbye! Thanks for the conversation!")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
