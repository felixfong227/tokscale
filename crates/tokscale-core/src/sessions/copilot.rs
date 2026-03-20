//! GitHub Copilot agent debug log parser.
//!
//! Parses JSONL files from VS Code workspaceStorage debug-log directories:
//! `.../GitHub.copilot-chat/debug-logs/<session-id>/*.jsonl`

use super::utils::{extract_i64, extract_string, file_modified_timestamp_ms, parse_timestamp_value};
use super::UnifiedMessage;
use crate::provider_identity::{canonical_provider, inferred_provider_from_model};
use crate::TokenBreakdown;
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub fn parse_copilot_file(path: &Path) -> Vec<UnifiedMessage> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(_) => return Vec::new(),
    };

    let fallback_timestamp = file_modified_timestamp_ms(path);
    let fallback_session_id = path
        .parent()
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("unknown")
        .to_string();
    let agent = agent_from_path(path);

    let reader = BufReader::new(file);
    let mut messages = Vec::with_capacity(32);

    for line in reader.lines() {
        let line = match line {
            Ok(line) => line,
            Err(_) => continue,
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut bytes = trimmed.as_bytes().to_vec();
        let value: Value = match simd_json::from_slice(&mut bytes) {
            Ok(value) => value,
            Err(_) => continue,
        };

        if value.get("type").and_then(Value::as_str) != Some("llm_request") {
            continue;
        }

        let attrs = value.get("attrs");
        let model = extract_string(attrs.and_then(|attrs| attrs.get("model")))
            .or_else(|| {
                value.get("name")
                    .and_then(Value::as_str)
                    .and_then(model_from_name)
            })
            .unwrap_or_else(|| "unknown".to_string());

        let provider = extract_string(attrs.and_then(|attrs| attrs.get("provider")))
            .and_then(|raw| canonical_provider(&raw).or(Some(raw)))
            .or_else(|| inferred_provider_from_model(&model).map(str::to_string))
            .unwrap_or_else(|| "github_copilot".to_string());

        let input = extract_i64(attrs.and_then(|attrs| attrs.get("inputTokens")))
            .unwrap_or(0)
            .max(0);
        let output = extract_i64(attrs.and_then(|attrs| attrs.get("outputTokens")))
            .unwrap_or(0)
            .max(0);
        let cache_read = extract_i64(
            attrs
                .and_then(|attrs| attrs.get("cachedTokens"))
                .or_else(|| attrs.and_then(|attrs| attrs.get("cacheReadTokens"))),
        )
        .unwrap_or(0)
        .max(0);
        let cache_write = extract_i64(attrs.and_then(|attrs| attrs.get("cacheWriteTokens")))
            .unwrap_or(0)
            .max(0);
        let reasoning = extract_i64(
            attrs
                .and_then(|attrs| attrs.get("reasoningTokens"))
                .or_else(|| attrs.and_then(|attrs| attrs.get("thoughtTokens"))),
        )
        .unwrap_or(0)
        .max(0);

        if input + output + cache_read + cache_write + reasoning == 0 {
            continue;
        }

        let timestamp = value
            .get("ts")
            .and_then(parse_timestamp_value)
            .unwrap_or(fallback_timestamp);
        let session_id =
            extract_string(value.get("sid")).unwrap_or_else(|| fallback_session_id.clone());

        messages.push(UnifiedMessage::new_with_agent(
            "copilot",
            model,
            provider,
            session_id,
            timestamp,
            TokenBreakdown {
                input,
                output,
                cache_read,
                cache_write,
                reasoning,
            },
            0.0,
            agent.clone(),
        ));
    }

    messages
}

fn model_from_name(name: &str) -> Option<String> {
    let (_, model) = name.split_once(':')?;
    let model = model.trim();
    if model.is_empty() {
        None
    } else {
        Some(model.to_string())
    }
}

fn agent_from_path(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    if stem == "main" || stem.starts_with("title-") {
        return None;
    }

    let rest = stem.strip_prefix("runSubagent-")?;
    let agent = strip_uuid_suffix(rest).trim();
    if agent.is_empty() {
        None
    } else {
        Some(agent.to_string())
    }
}

fn strip_uuid_suffix(value: &str) -> &str {
    if value.len() <= 37 {
        return value;
    }

    let suffix_start = value.len() - 36;
    if value.as_bytes()[suffix_start - 1] != b'-' {
        return value;
    }

    let candidate = &value[suffix_start..];
    if is_uuid(candidate) {
        &value[..suffix_start - 1]
    } else {
        value
    }
}

fn is_uuid(value: &str) -> bool {
    if value.len() != 36 {
        return false;
    }

    for (index, ch) in value.chars().enumerate() {
        let should_be_hyphen = matches!(index, 8 | 13 | 18 | 23);
        if should_be_hyphen {
            if ch != '-' {
                return false;
            }
        } else if !ch.is_ascii_hexdigit() {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use tempfile::TempDir;

    fn write_log(dir: &TempDir, relative: &str, content: &str) -> PathBuf {
        let path = dir.path().join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, content).unwrap();
        path
    }

    #[test]
    fn test_parse_copilot_file_parses_llm_request_events() {
        let dir = TempDir::new().unwrap();
        let path = write_log(
            &dir,
            "debug-logs/session-123/main.jsonl",
            concat!(
                "{\"ts\":1773990037678,\"sid\":\"session-123\",\"type\":\"discovery\",\"name\":\"Load Skills\",\"attrs\":{}}\n",
                "{\"ts\":1773990040731,\"sid\":\"session-123\",\"type\":\"llm_request\",\"name\":\"chat:gpt-5.4\",\"attrs\":{\"model\":\"gpt-5.4\",\"inputTokens\":27376,\"outputTokens\":689,\"ttft\":5921}}\n"
            ),
        );

        let messages = parse_copilot_file(&path);

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].client, "copilot");
        assert_eq!(messages[0].session_id, "session-123");
        assert_eq!(messages[0].model_id, "gpt-5.4");
        assert_eq!(messages[0].provider_id, "openai");
        assert_eq!(messages[0].tokens.input, 27376);
        assert_eq!(messages[0].tokens.output, 689);
        assert_eq!(messages[0].agent, None);
    }

    #[test]
    fn test_parse_copilot_file_infers_model_and_agent_from_child_log_name() {
        let dir = TempDir::new().unwrap();
        let path = write_log(
            &dir,
            "debug-logs/session-123/runSubagent-Explore-12345678-1234-1234-1234-123456789abc.jsonl",
            "{\"ts\":1773990065243,\"sid\":\"child-session\",\"type\":\"llm_request\",\"name\":\"chat:claude-sonnet-4\",\"attrs\":{\"inputTokens\":15473,\"outputTokens\":470}}\n",
        );

        let messages = parse_copilot_file(&path);

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].session_id, "child-session");
        assert_eq!(messages[0].model_id, "claude-sonnet-4");
        assert_eq!(messages[0].provider_id, "anthropic");
        assert_eq!(messages[0].agent.as_deref(), Some("Explore"));
    }

    #[test]
    fn test_parse_copilot_file_falls_back_to_parent_directory_session_id() {
        let dir = TempDir::new().unwrap();
        let path = write_log(
            &dir,
            "debug-logs/session-fallback/main.jsonl",
            "{\"type\":\"llm_request\",\"name\":\"chat:gpt-4o\",\"attrs\":{\"inputTokens\":10,\"outputTokens\":5,\"cachedTokens\":2,\"reasoningTokens\":1}}\n",
        );

        let messages = parse_copilot_file(&path);

        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].session_id, "session-fallback");
        assert_eq!(messages[0].model_id, "gpt-4o");
        assert_eq!(messages[0].provider_id, "openai");
        assert_eq!(messages[0].tokens.cache_read, 2);
        assert_eq!(messages[0].tokens.reasoning, 1);
    }
}