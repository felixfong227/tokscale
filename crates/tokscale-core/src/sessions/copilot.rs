//! VS Code GitHub Copilot chat replay parser
//!
//! Parses exported `.chatreplay.json` files from the Copilot Agent Debug view.

use super::utils::{file_modified_timestamp_ms, parse_timestamp_str};
use super::UnifiedMessage;
use crate::TokenBreakdown;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct ChatReplayExport {
    #[serde(rename = "exportedAt")]
    exported_at: Option<String>,
    #[serde(default)]
    prompts: Vec<ExportedPrompt>,
}

#[derive(Debug, Deserialize)]
struct ExportedPrompt {
    #[serde(default)]
    logs: Vec<ExportedLogEntry>,
}

#[derive(Debug, Deserialize)]
struct ExportedLogEntry {
    id: Option<String>,
    metadata: Option<ExportedLogMetadata>,
    time: Option<String>,
    timestamp: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExportedLogMetadata {
    model: Option<String>,
    #[serde(rename = "startTime")]
    start_time: Option<String>,
    #[serde(rename = "endTime")]
    end_time: Option<String>,
    usage: Option<ExportedUsage>,
}

#[derive(Debug, Deserialize)]
struct ExportedUsage {
    prompt_tokens: Option<i64>,
    completion_tokens: Option<i64>,
    total_tokens: Option<i64>,
}

pub fn parse_copilot_file(path: &Path) -> Vec<UnifiedMessage> {
    let data = match std::fs::read(path) {
        Ok(data) => data,
        Err(_) => return Vec::new(),
    };

    let mut bytes = data;
    let export: ChatReplayExport = match simd_json::from_slice(&mut bytes) {
        Ok(export) => export,
        Err(_) => return Vec::new(),
    };

    let export_timestamp = export
        .exported_at
        .as_deref()
        .and_then(parse_timestamp_str)
        .unwrap_or_else(|| file_modified_timestamp_ms(path));
    let session_id = extract_session_id(path);

    export
        .prompts
        .into_iter()
        .flat_map(|prompt| {
            prompt.logs.into_iter().filter_map(|log| {
                let metadata = log.metadata?;
                let usage = metadata.usage?;

                let input = usage.prompt_tokens.unwrap_or(0).max(0);
                let output = usage.completion_tokens.unwrap_or(0).max(0);
                let total = usage.total_tokens.unwrap_or(0).max(0);
                if input == 0 && output == 0 && total == 0 {
                    return None;
                }

                let model_id = metadata.model.unwrap_or_else(|| "unknown".to_string());
                let timestamp = metadata
                    .start_time
                    .as_deref()
                    .or(metadata.end_time.as_deref())
                    .or(log.time.as_deref())
                    .or(log.timestamp.as_deref())
                    .and_then(parse_timestamp_str)
                    .unwrap_or(export_timestamp);

                let dedup_key = log.id.map(|id| format!("{}:{}", session_id, id));

                Some(UnifiedMessage::new_with_dedup(
                    "copilot",
                    model_id,
                    "github-copilot",
                    session_id.clone(),
                    timestamp,
                    TokenBreakdown {
                        input,
                        output,
                        cache_read: 0,
                        cache_write: 0,
                        reasoning: 0,
                    },
                    0.0,
                    dedup_key,
                ))
            })
        })
        .collect()
}

fn extract_session_id(path: &Path) -> String {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    file_name
        .strip_suffix(".chatreplay.json")
        .unwrap_or(file_name)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_chatreplay(dir: &TempDir, file_name: &str, content: &str) -> std::path::PathBuf {
        let path = dir.path().join(file_name);
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn test_parse_copilot_chatreplay_usage() {
        let dir = TempDir::new().unwrap();
        let path = write_chatreplay(
            &dir,
            "copilot_all_prompts_2026-03-07T11-23-48.chatreplay.json",
            r#"{
  "exportedAt": "2026-03-07T11:23:48Z",
  "totalPrompts": 1,
  "totalLogEntries": 2,
  "prompts": [
    {
      "prompt": "Explain this issue",
      "logCount": 2,
      "logs": [
        {
          "id": "log-1",
          "kind": "request",
          "metadata": {
            "model": "claude-sonnet-4.5",
            "startTime": "2026-03-07T11:20:00Z",
            "usage": {
              "prompt_tokens": 123,
              "completion_tokens": 45,
              "total_tokens": 168
            }
          }
        },
        {
          "id": "log-2",
          "kind": "request",
          "metadata": {
            "model": "gpt-4.1",
            "usage": {
              "prompt_tokens": 10,
              "completion_tokens": 5,
              "total_tokens": 15
            }
          },
          "time": "2026-03-07T11:21:00Z"
        }
      ]
    }
  ]
}"#,
        );

        let messages = parse_copilot_file(&path);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].client, "copilot");
        assert_eq!(messages[0].provider_id, "github-copilot");
        assert_eq!(messages[0].model_id, "claude-sonnet-4.5");
        assert_eq!(messages[0].tokens.input, 123);
        assert_eq!(messages[0].tokens.output, 45);
        assert_eq!(
            messages[0].session_id,
            "copilot_all_prompts_2026-03-07T11-23-48"
        );
        assert!(messages[0].dedup_key.as_deref().is_some());
        assert_eq!(messages[1].model_id, "gpt-4.1");
        assert_eq!(messages[1].tokens.input, 10);
        assert_eq!(messages[1].tokens.output, 5);
    }

    #[test]
    fn test_parse_copilot_skips_logs_without_usage() {
        let dir = TempDir::new().unwrap();
        let path = write_chatreplay(
            &dir,
            "copilot.chatreplay.json",
            r#"{
  "exportedAt": "2026-03-07T11:23:48Z",
  "prompts": [
    {
      "prompt": "Explain this issue",
      "logCount": 2,
      "logs": [
        {
          "id": "log-1",
          "kind": "request",
          "metadata": {
            "model": "claude-sonnet-4.5"
          }
        },
        {
          "id": "log-2",
          "kind": "request",
          "metadata": {
            "model": "claude-sonnet-4.5",
            "usage": {
              "prompt_tokens": 0,
              "completion_tokens": 0,
              "total_tokens": 0
            }
          }
        }
      ]
    }
  ]
}"#,
        );

        let messages = parse_copilot_file(&path);
        assert!(messages.is_empty());
    }
}
